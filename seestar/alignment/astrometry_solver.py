"""
Module pour gérer l'interaction avec les solveurs astrométriques.

Deux chemins de résolution seulement : ZeSolver (optionnel, prioritaire) et
ASTAP (fallback autonome). Les solveurs ANSVR et Astrometry.net (web) ont été
retirés de Zsss ; ZeSolver les gère via ses stratégies internes.
"""
import os
import re
import numpy as np
import warnings
import time
import traceback
import subprocess  # Pour appeler les solveurs locaux
import logging
import platform
from seestar.core.solver_config import get_astap_default_search_radius
try:  # Allow running as a standalone module in tests
    from ..core.image_processing import sanitize_header_for_wcs
except Exception:  # pragma: no cover
    try:
        from seestar.core.image_processing import sanitize_header_for_wcs
    except Exception:  # pragma: no cover
        def sanitize_header_for_wcs(hdr):  # type: ignore
            return hdr

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Default search radius in degrees used by ASTAP when no value is provided
# through solver settings. Loaded from ``seestar.core.solver_config`` so tests and
# documentation stay in sync with application defaults.
ASTAP_DEFAULT_SEARCH_RADIUS = get_astap_default_search_radius()


def resolve_astap_executable(path: str) -> str:
    """Return the actual ASTAP binary for the provided path.

    On macOS the application is typically distributed as an ``.app`` bundle.
    In this case the real executable resides under
    ``<app>/Contents/MacOS/astap`` (or ``ASTAP``).  This helper resolves the
    path automatically while leaving paths on other systems untouched.
    """
    if (
        path
        and path.lower().endswith(".app")
        and os.path.isdir(path)
        and platform.system() == "Darwin"
    ):
        candidate = os.path.join(path, "Contents", "MacOS", "astap")
        if os.path.isfile(candidate):
            return candidate
        candidate_upper = os.path.join(path, "Contents", "MacOS", "ASTAP")
        if os.path.isfile(candidate_upper):
            return candidate_upper
    return path

# --- Dépendances Astropy ---
_ASTROPY_AVAILABLE = False

try:
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyError
    from astropy.wcs import WCS, FITSFixedWarning
    from astropy.utils.exceptions import AstropyWarning
    _ASTROPY_AVAILABLE = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)  # Pour d'autres avertissements astropy
except ImportError:
    logger.error(
        "ERREUR CRITIQUE [AstrometrySolverModule]: Astropy non installée. Le module ne peut fonctionner.")


try:
    from seestar.alignment.zesolver_adapter import ZeSolverAdapter
except Exception:  # pragma: no cover - l'adaptateur optionnel ne doit jamais casser le solver
    ZeSolverAdapter = None

# Préférences solver obsolètes désormais gérées par ZeSolver (qui internalise les
# stratégies Astrometry.net). Migrées avec un WARN unique.
_LEGACY_PREFERENCES = ("ansvr", "astrometry")
_legacy_preference_warned = False


def _canonicalize_wcs_scale(wcs_obj):
    """Normalize a WCS so the pixel scale is encoded exactly once.

    ASTAP (recent builds) writes ``.wcs`` sidecars that carry the pixel scale
    simultaneously in the CD matrix, in ``CDELT`` and in ``PC``.  astropy keeps
    all three representations, and wcslib then applies ``PC x CDELT`` so the
    effective pixel scale becomes ``scale^2`` (e.g. ~0.0016 arcsec/pix for a
    genuine 2.37 arcsec/pix Seestar solution).  Every pixel->world transform
    is then corrupted (the whole field collapses onto a few pixels) and
    ``proj_plane_pixel_scales`` reports the bogus ``scale^2`` value, which
    triggers the "Reference WCS pixel scale ... outside [0.1, 30.0]; clipping"
    warning.

    When a CD matrix is present we rebuild the WCS from it so that ``PC`` is a
    dimensionless rotation and ``CDELT`` carries the scale: the scale is then
    encoded exactly once, transforms are correct, and ``to_header`` round-trips
    cleanly (no double encoding).  WCSes that already use a single encoding
    (``cd`` absent) are left untouched.

    Returns the (possibly mutated) WCS object.
    """
    if wcs_obj is None:
        return wcs_obj
    try:
        cd = np.asarray(wcs_obj.wcs.cd, dtype=float)
    except AttributeError:
        return wcs_obj
    if cd.ndim != 2 or cd.shape != (2, 2):
        return wcs_obj
    col_norms = np.sqrt(np.sum(cd ** 2, axis=0))
    if not (np.all(np.isfinite(col_norms)) and np.all(col_norms > 0)):
        return wcs_obj
    # Drop the CD matrix *before* writing PC/CDELT: assigning ``cdelt`` while
    # ``cd`` is present makes wcslib emit ``RuntimeWarning: cdelt will be
    # ignored since cd is present``.  ``del wcs.cd`` is the documented wcslib
    # way to switch to the canonical PC + CDELT representation, leaving the WCS
    # truly single-encoded (``has_cd()`` becomes False).
    del wcs_obj.wcs.cd
    # cd == pc @ diag(cdelt): keep the transform exact, encode the scale once.
    wcs_obj.wcs.pc = cd / col_norms[np.newaxis, :]
    wcs_obj.wcs.cdelt = col_norms
    return wcs_obj


def _strip_redundant_scale_keywords(hdr):
    """Remove CDELT/PC (and CROTA2) when a CD matrix is present.

    ASTAP writes ``.wcs`` sidecars carrying the pixel scale in CD, CDELT and PC
    simultaneously.  FITS gives ``CDi_j`` precedence over ``PCi_j`` + ``CDELTi``,
    so the CDELT/PC forms are redundant whenever CD is present.  Dropping them
    up-front guarantees wcslib never sees the ambiguous triple encoding at WCS
    construction time; the scale is then re-encoded exactly once by
    :func:`_canonicalize_wcs_scale`.  Headers without a CD matrix are returned
    unchanged.
    """
    if not any(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        return hdr
    for k in list(hdr.keys()):
        if k in ("CDELT1", "CDELT2", "CROTA2"):
            del hdr[k]
        elif re.match(r"^PC\d_\d$", k):
            del hdr[k]
    return hdr


def _sanitize_astap_wcs_text(txt: str) -> tuple[str, int, int]:
    """
    Sanitize raw ASTAP ``.wcs`` text before parsing.

    Parameters
    ----------
    txt : str
        Raw text content of the ``.wcs`` sidecar produced by ASTAP.

    Returns
    -------
    tuple
        ``(clean_text, modified, dropped)`` where ``clean_text`` is the
        sanitized header, ``modified`` counts ``CONTINUE`` lines that were
        adjusted and ``dropped`` counts invalid ``CONTINUE`` lines removed.
    """

    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    out: list[str] = []
    modified = 0
    dropped = 0

    for l in lines:
        if l.lstrip().startswith('CONTINUE'):
            m = re.match(r'^CONTINUE\s+(.*)$', l.strip())
            if not m:
                dropped += 1
                continue
            payload = m.group(1).strip()
            if not (payload.startswith("'") or payload.startswith('"')):
                payload = payload.replace('"', '\\"')
                payload = f'"{payload}"'
                modified += 1
            out.append(f"CONTINUE {payload}")
        else:
            out.append(l)

    return "\n".join(out) + "\n", modified, dropped


class AstrometrySolver:
    """
    Classe pour orchestrer la résolution astrométrique en utilisant différents solveurs.

    Deux chemins de résolution seulement : ZeSolver (optionnel) et ASTAP (fallback).
    """
    #: Classe d'adaptateur ZeSolver (importée au niveau module — jamais le package
    #: ``zesolver`` lui-même). L'*instance* est créée paresseusement au premier usage.
    _zesolver_adapter_class = ZeSolverAdapter

    def __init__(self, progress_callback=None, verbose=None):
        """
        Initialise le solveur.
        Args:
            progress_callback (callable, optional): Callback pour les messages de progression.
        """
        self.progress_callback = progress_callback
        if verbose is None:
            _v_env = os.getenv("SEESTAR_VERBOSE", "")
            verbose = str(_v_env).lower() in ("1", "true", "yes")
        self.verbose = verbose
        self.logger = logger
        if not _ASTROPY_AVAILABLE:
            self._log("ERREUR CRITIQUE: Astropy n'est pas disponible. AstrometrySolver ne peut fonctionner.", "ERROR")
            raise ImportError("Astropy est requis pour AstrometrySolver.")
        # Valeurs par défaut GLOBALES pour l'estimation d'échelle si FITS incomplet
        # Ces valeurs seront écrasées par celles des 'settings' dans la méthode solve() si fournies.
        self.default_pixel_size_um_for_cfg = 2.4  # Valeur Seestar S50 par défaut
        self.default_focal_length_mm_for_cfg = 250.0 # Valeur Seestar S50 par défaut
        self._zesolver_adapter = None

    def _extract_scale_arcsec(self, wcs_obj):
        """Return pixel scale in arcsec/pixel from a WCS object."""
        if wcs_obj and hasattr(wcs_obj, "pixel_scale_matrix"):
            try:
                return float(np.sqrt(np.abs(np.linalg.det(wcs_obj.pixel_scale_matrix))) * 3600.0)
            except Exception:
                return float("nan")
        return float("nan")


    def _log(self, message, level="INFO"):
        prefix_map = {
            "INFO": "   [AstrometrySolver]",
            "WARN": "   ⚠️ [AstrometrySolver WARN]",
            "ERROR": "   ❌ [AstrometrySolver ERROR]",
            "DEBUG": "      [AstrometrySolver DEBUG]"
        }
        level_upper = str(level).upper()
        if level_upper == "DEBUG" and not self.verbose:
            return

        prefix = prefix_map.get(level_upper, prefix_map["INFO"])
        full_msg = f"{prefix} {message}"

        if self.progress_callback and callable(self.progress_callback):
            try:
                self.progress_callback(full_msg, None)
            except Exception:
                self.logger.log(logging.ERROR, "Progress callback failed for log message")

        log_level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
        }.get(level_upper, logging.INFO)
        self.logger.log(log_level, full_msg)


    def solve(
        self,
        image_path,
        fits_header,
        settings,
        update_header_with_solution=True,
        is_boring_stack_disk_mode=False,
        *,
        batch_size=None,
        final_combine=None,
    ):
        """
        Tente de résoudre le WCS d'une image en utilisant la stratégie configurée.

        Deux chemins de résolution existent : ZeSolver (optionnel, prioritaire
        lorsqu'il est installé/compatible/configuré) et ASTAP (fallback autonome).
        Les anciens solveurs ANSVR et Astrometry.net (web) sont désormais gérés
        exclusivement par ZeSolver via ses stratégies internes.

        Args:
            image_path (str): Chemin vers le fichier image à résoudre.
            fits_header (fits.Header): Header FITS de l'image.
            settings (dict): Dictionnaire contenant la configuration des solveurs.
                             Clés attendues: 'local_solver_preference'
                             ("none", "astap", "zesolver"; les valeurs obsolètes
                             "ansvr"/"astrometry" sont migrées vers "zesolver"),
                             'astap_path', 'astap_data_dir', 'astap_search_radius',
                             'astap_downsample', 'astap_sensitivity',
                             'astap_timeout_sec', 'scale_est_arcsec_per_pix',
                             'scale_tolerance_percent', 'use_radec_hints'.
            update_header_with_solution (bool): Si True, met à jour ``fits_header``
                avec la solution.
            is_boring_stack_disk_mode (bool): True uniquement pour le pipeline
                disque ``batch_size=1``.

        Returns:
            astropy.wcs.WCS or None: Objet WCS si succès, None si échec.
        """
        if (
            batch_size == 1
            and str(final_combine).lower()
            in {"reproject_and_coadd", "reproject", "coadd"}
        ):
            norm = image_path.replace("\\", "/").lower()
            if "/aligned_tmp/" in norm or "/classic_batch_outputs/" in norm:
                logger.info(
                    "[AstrometrySolver] Skip solving intermediate aligned/batch in BS=1+Reproject mode: %s",
                    image_path,
                )
                return None

        self._log(
            f"Début résolution pour: {os.path.basename(image_path)} (Utilisation de 'local_solver_preference')",
            "INFO",
        )
        wcs_solution = None

        # --- Récupération des paramètres depuis le dictionnaire settings ---
        solver_preference = settings.get('local_solver_preference', "none")
        scale_est = settings.get('scale_est_arcsec_per_pix', None)
        scale_tol = settings.get('scale_tolerance_percent', 20)

        astap_exe = settings.get('astap_path', "")
        astap_data = settings.get('astap_data_dir', None)
        astap_search_radius_from_settings = settings.get(
            'astap_search_radius', ASTAP_DEFAULT_SEARCH_RADIUS
        )
        astap_downsample_val = settings.get('astap_downsample', 2)
        astap_sensitivity_val = settings.get('astap_sensitivity', 100)
        astap_timeout = settings.get('astap_timeout_sec', 120)
        use_radec_hints = settings.get('use_radec_hints', False)

        # Migration: les préférences obsolètes pointent vers ZeSolver (qui possède
        # désormais les stratégies Astrometry.net).
        if solver_preference in ("ansvr", "astrometry"):
            solver_preference = self._migrate_legacy_preference(solver_preference)

        self._log(f"Solver preference: '{solver_preference}'", "DEBUG")
        self._log(
            f"ASTAP Exe: '{astap_exe}', Data: '{astap_data}', Radius: {astap_search_radius_from_settings}, Timeout: {astap_timeout}",
            "DEBUG",
        )
        self._log(f"Use RA/DEC hints: {use_radec_hints}", "DEBUG")

        # --- ZeSolver (optionnel, primaire) ---
        if solver_preference == "zesolver":
            wcs_solution, allow_fallback = self._try_solve_zesolver(
                image_path,
                fits_header,
                settings,
                update_header_with_solution,
            )
            if wcs_solution is not None:
                return wcs_solution
            if not allow_fallback:
                self._log("ZeSolver: résolution annulée; aucun fallback ASTAP.", "INFO")
                return None
            self._log("ZeSolver: échec/indisponible; tentative ASTAP en fallback.", "WARN")

        # --- ASTAP (primaire pour "astap", fallback pour "zesolver") ---
        if solver_preference in ("astap", "zesolver"):
            astap_exe_resolved = resolve_astap_executable(astap_exe)
            if astap_exe_resolved and os.path.isfile(astap_exe_resolved):
                if astap_exe_resolved != astap_exe:
                    self._log(
                        f"ASTAP: bundle detected, using executable '{astap_exe_resolved}'.",
                        "DEBUG",
                    )
                if solver_preference == "astap":
                    self._log("Priorité au solveur local: ASTAP.", "INFO")
                else:
                    self._log("Fallback vers le solveur local ASTAP.", "INFO")
                t0 = time.time()
                wcs_solution = self._try_solve_astap(
                    image_path,
                    fits_header,
                    astap_exe_resolved,
                    astap_data,
                    astap_search_radius_from_settings,
                    scale_est,
                    scale_tol,
                    astap_timeout,
                    update_header_with_solution,
                    astap_downsample_val,
                    astap_sensitivity_val,
                    use_radec_hints,
                    is_boring_stack_disk_mode=is_boring_stack_disk_mode,
                )
                if wcs_solution:
                    dt = time.time() - t0
                    scale = self._extract_scale_arcsec(wcs_solution)
                    self._log(
                        f"🔭 [Solver] ASTAP OK  –  scale {scale:.2f}\"/px  RMS 0.00″  (elapsed {dt:.1f}s)",
                        "INFO",
                    )
                    return wcs_solution
                else:
                    self._log("ASTAP a échoué ou n'a pas trouvé de solution.", "WARN")
            else:
                self._log(
                    f"ASTAP sélectionné mais chemin exécutable '{astap_exe}' invalide ou non fourni.",
                    "WARN",
                )
        elif solver_preference == "none":
            self._log(
                "Aucun solveur configuré (local_solver_preference='none'). Aucune résolution tentée.",
                "INFO",
            )

        if not wcs_solution:
            self._log(
                f"Aucune solution astrométrique trouvée pour {os.path.basename(image_path)} après toutes les tentatives configurées.",
                "WARN",
            )

        return None

    def _migrate_legacy_preference(self, preference):
        """Map a legacy solver preference onto ``zesolver`` (one-time WARN)."""
        global _legacy_preference_warned
        if not _legacy_preference_warned:
            _legacy_preference_warned = True
            self._log(
                f"Préférence solver obsolète '{preference}' migrée vers 'zesolver' "
                "(ZeSolver gère désormais les stratégies Astrometry.net).",
                "WARN",
            )
        return "zesolver"

    def _get_zesolver_adapter(self):
        """Return the lazily-created ZeSolver adapter (one per solver instance).

        The adapter *instance* is created on first use only; the class is
        imported at module load (it never imports ``zesolver`` itself). Returns
        ``None`` when the adapter is unavailable (import failure).
        """
        if self._zesolver_adapter is None:
            adapter_cls = self._zesolver_adapter_class
            if adapter_cls is None:
                self._log("ZeSolver: classe d'adaptateur indisponible (import échoué).", "WARN")
                return None
            self._zesolver_adapter = adapter_cls()
        return self._zesolver_adapter

    def _try_solve_zesolver(
        self,
        image_path,
        fits_header,
        settings,
        update_header_with_solution,
    ):
        """Run the optional ZeSolver path and map its outcome onto a WCS.

        Returns ``(wcs, allow_fallback)``:
          * ``wcs`` is the solved WCS (or the pre-existing WCS on SKIPPED, or
            ``None`` when nothing usable was produced).
          * ``allow_fallback`` is ``False`` only when the user cancelled the
            solve (the caller then returns ``None`` without ASTAP).
        """
        adapter = self._get_zesolver_adapter()
        if adapter is None:
            self._log("ZeSolver indisponible (adaptateur absent). Fallback ASTAP.", "WARN")
            return None, True

        try:
            outcome = adapter.solve(
                image_fits_path=image_path,
                fits_header=fits_header,
                settings=settings,
                progress_callback=self.progress_callback,
                log=self._log,
            )
        except Exception as exc:
            self._log(
                f"ZeSolver: exception inattendue {type(exc).__name__}: {exc}",
                "WARN",
            )
            return None, True

        if outcome is None:
            self._log("ZeSolver: résultat nul. Fallback ASTAP.", "WARN")
            return None, True

        status = getattr(outcome, "status", None)
        status_value = getattr(status, "value", status)

        if status_value == "solved":
            wcs = getattr(outcome, "wcs", None)
            if wcs is None or not getattr(wcs, "is_celestial", False):
                self._log("ZeSolver: SOLVED mais WCS absent/non céleste. Fallback ASTAP.", "WARN")
                return None, True
            if update_header_with_solution and fits_header is not None:
                self._update_fits_header_with_wcs(
                    fits_header, wcs, solver_name="ZeSolver"
                )
            scale = self._extract_scale_arcsec(wcs)
            self._log(f"🔭 [Solver] ZeSolver OK  –  scale {scale:.2f}\"/px", "INFO")
            return wcs, False

        if status_value == "skipped":
            existing = self._wcs_from_header(fits_header)
            if existing is not None:
                self._log(
                    "ZeSolver: SKIPPED (WCS déjà présent). Utilisation du WCS existant.",
                    "INFO",
                )
                return existing, False
            self._log(
                "ZeSolver: SKIPPED mais aucun WCS céleste dans le header. Fallback ASTAP.",
                "WARN",
            )
            return None, True

        if status_value == "cancelled":
            self._log("ZeSolver: résolution annulée par l'utilisateur.", "INFO")
            return None, False

        if status_value == "unavailable":
            msg = getattr(outcome, "message", None) or "absent/incompatible"
            self._log(f"ZeSolver indisponible ({msg}). Fallback ASTAP.", "WARN")
            return None, True

        self._log(f"ZeSolver: échec ({status_value}). Fallback ASTAP.", "WARN")
        return None, True

    @staticmethod
    def _wcs_from_header(header):
        """Return a celestial WCS parsed from a FITS header, or ``None``."""
        if header is None:
            return None
        try:
            wcs = WCS(header, naxis=2, relax=True)
            if wcs.is_celestial:
                return wcs
        except Exception:
            pass
        return None

    def _derive_pixel_scale_from_header(self, header):
        """Return pixel scale (arcsec/pix) derived from FITS header if possible."""
        if not header:
            self._log("Pixel scale derivation: header None", "DEBUG")
            return None

        pixel_um = None
        focal_mm = None

        for key in ("XPIXSZ", "PIXSIZE1"):
            if key in header:
                try:
                    val = float(header[key])
                    if val > 0:
                        pixel_um = val
                        break
                except Exception:
                    pass

        if "FOCALLEN" in header:
            try:
                val = float(header["FOCALLEN"])
                if val > 0:
                    focal_mm = val
            except Exception:
                pass

        if pixel_um and focal_mm:
            scale = (pixel_um / focal_mm) * 206.265
            self._log(
                f"Pixel scale derived from header: {scale:.3f} arcsec/pix (pix={pixel_um}µm, focal={focal_mm}mm)",
                "DEBUG",
            )
            return scale

        self._log("Pixel scale derivation failed due to missing keywords", "DEBUG")
        return None


    def _try_solve_astap(
        self,
        image_path,
        fits_header,
        astap_exe_path,
        astap_data_dir,
        astap_search_radius_deg,
        scale_est_arcsec_per_pix_from_solver_UNUSED,
        scale_tolerance_percent_UNUSED,
        timeout_sec,
        update_header_with_solution,
        astap_downsample=2,
        astap_sensitivity=100,
        use_radec_hints=False,
        is_boring_stack_disk_mode=False,
    ):
        self._log(f"Entering _try_solve_astap for {os.path.basename(image_path)}", "DEBUG")
        self._log(f"ASTAP: Début résolution pour {os.path.basename(image_path)}", "INFO")

        image_dir = os.path.dirname(image_path)
        base_image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

        # --- NOMS DES FICHIERS ATTENDUS ---
        expected_wcs_file = os.path.join(image_dir, base_image_name_no_ext + ".wcs")
        expected_ini_file = os.path.join(image_dir, base_image_name_no_ext + ".ini")
        # Le fichier .log généré par l'option -log d'ASTAP aura le même nom de base que l'image
        astap_log_file_generated = os.path.join(image_dir, base_image_name_no_ext + ".log")

        files_to_cleanup = [expected_wcs_file, expected_ini_file, astap_log_file_generated]

        # --- NETTOYAGE PRÉ-EXÉCUTION ---
        self._log(f"ASTAP: Nettoyage pré-exécution des fichiers temporaires potentiels...", "DEBUG")
        for f_to_clean_pre in files_to_cleanup:
            if os.path.exists(f_to_clean_pre):
                try:
                    os.remove(f_to_clean_pre)
                    self._log(f"ASTAP: Ancien fichier '{os.path.basename(f_to_clean_pre)}' supprimé avant exécution.", "DEBUG")
                except Exception as e_del_pre:
                    self._log(f"ASTAP: Avertissement - Échec suppression pré-exécution de '{os.path.basename(f_to_clean_pre)}': {e_del_pre}", "WARN")
        # --- FIN NETTOYAGE PRÉ-EXÉCUTION ---

        cmd = [astap_exe_path, "-f", image_path, "-log"] # Option -log pour générer le .log
        if astap_data_dir and os.path.isdir(astap_data_dir):
            cmd.extend(["-d", astap_data_dir])

        # Options de résolution (z, sens)
        cmd.extend(["-z", str(astap_downsample)])  # Downsample configurable
        cmd.extend(["-sens", str(astap_sensitivity)])  # Détection configurable

        # Gestion du rayon de recherche
        # astap_search_radius_deg est la valeur float reçue de settings
        if astap_search_radius_deg is not None and astap_search_radius_deg > 0:
            # ASTAP attend un rayon en degrés, ce que nous avons.
            # Si RA/DEC sont aussi fournis, ce rayon est centré.
            # Si pas de RA/DEC, ASTAP utilise ce rayon autour du centre de l'image (s'il ne trouve pas avec -fov 0).
            # L'option -fov 0 demande à ASTAP d'estimer lui-même le champ.
            # On peut soit utiliser -fov 0 (et laisser ASTAP décider), soit passer -r si on a une bonne estimation.
            # Pour l'instant, on passe -r si fourni, sinon on laisse ASTAP gérer.
            # Le comportement exact de -r sans -ra -dec est à confirmer via les logs ASTAP.
            # Le log d'ASTAP devrait indiquer "Search an area of X degrees around image center"
            radius_str = f"{float(astap_search_radius_deg):.2f}"
            cmd.extend(["-r", radius_str])
            self._log(f"ASTAP: Utilisation rayon de recherche: {radius_str}°", "DEBUG")
        else:
            # Si astap_search_radius_deg est 0 ou non fourni, ASTAP utilisera -fov 0
            # ce qui est généralement recommandé pour une recherche "aveugle".
            cmd.extend(["-fov", "0"])
            self._log(f"ASTAP: Utilisation -fov 0 (recherche automatique du champ).", "DEBUG")

        # Provide RA/DEC hints if enabled and present in the FITS header
        ra_hint = None
        dec_hint = None
        if use_radec_hints and fits_header:
            ra_hint = fits_header.get('RA', fits_header.get('CRVAL1'))
            dec_hint = fits_header.get('DEC', fits_header.get('CRVAL2'))
        hints_status_msg = "désactivés"
        if use_radec_hints and isinstance(ra_hint, (int, float)) and isinstance(dec_hint, (int, float)):
            cmd.extend(["-ra", str(ra_hint), "-dec", str(dec_hint)])
            self._log(
                f"ASTAP: Hints RA={ra_hint} DEC={dec_hint} ajoutés à la commande.",
                "DEBUG",
            )
            hints_status_msg = f"utilisés -> RA={ra_hint} DEC={dec_hint}"
        elif use_radec_hints:
            hints_status_msg = "activés mais valeurs manquantes ou invalides"
        self._log(f"ASTAP: RA/DEC hints avant exécution: {hints_status_msg}", "DEBUG")

        # Determine pixel scale from header if possible
        pxscale = self._derive_pixel_scale_from_header(fits_header)
        if isinstance(pxscale, (int, float)) and 0.1 <= pxscale <= 50.0:
            cmd.extend(["-pxscale", f"{pxscale:.3f}"])
            self._log(f"ASTAP: Option -pxscale {pxscale:.3f} utilisée.", "DEBUG")
        else:
            if "-fov" not in cmd:
                cmd.extend(["-fov", "0"])
                self._log("ASTAP: Option -fov 0 ajoutée (échelle inconnue).", "DEBUG")
            else:
                self._log("ASTAP: Échelle inconnue mais -fov déjà spécifié.", "DEBUG")

        self._log(f"ASTAP: Commande finale: {' '.join(cmd)}", "DEBUG")
        wcs_object = None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False, cwd=image_dir)
            self._log(f"ASTAP: Code de retour: {result.returncode}", "DEBUG")
            if result.stdout: self._log(f"ASTAP stdout (premiers 500 caractères):\n{result.stdout[:500]}", "DEBUG")
            if result.stderr: self._log(f"ASTAP stderr (premiers 500 caractères):\n{result.stderr[:500]}", "DEBUG")

            if result.returncode == 0:
                img_shape_hw_for_wcs = None
                try:
                    with fits.open(image_path, memmap=False) as hdul_img_shape:
                        img_data_shape = hdul_img_shape[0].shape
                        if len(img_data_shape) >= 2:
                            img_shape_hw_for_wcs = img_data_shape[-2:]  # (H, W)
                        else:
                            raise ValueError(f"Shape image inattendue: {img_data_shape}")
                except Exception as e_shape:
                    self._log(
                        f"ASTAP: Erreur lecture shape image ('{image_path}') pour WCS parsing: {e_shape}. Utilisation fallback header.",
                        "WARN",
                    )
                    h_fallback = fits_header.get('NAXIS2', 1000) if fits_header else 1000
                    w_fallback = fits_header.get('NAXIS1', 1000) if fits_header else 1000
                    img_shape_hw_for_wcs = (int(h_fallback), int(w_fallback))

                if os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) > 0:
                    self._log(f"ASTAP: Résolution réussie. Fichier '{expected_wcs_file}' trouvé.", "INFO")
                    wcs_object = self._parse_wcs_file_content(
                        expected_wcs_file, img_shape_hw_for_wcs
                    )

                    if not (wcs_object and wcs_object.is_celestial) and is_boring_stack_disk_mode:
                        self._log(
                            "[ASTAP WCS] Using FITS header fallback (batch_size=1 path)",
                            "DEBUG",
                        )
                        try:
                            with fits.open(image_path, memmap=False) as hdul:
                                hdr_fits = hdul[0].header.copy()
                            for card in hdr_fits.cards:
                                if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                                    card.value = str(card.value)
                            sanitize_header_for_wcs(hdr_fits)
                            wcs_object = WCS(hdr_fits, naxis=2, relax=True)
                            assert wcs_object.is_celestial
                        except Exception as e_hdr:
                            self._log(
                                f"WCS parse failed from FITS header fallback: {e_hdr}",
                                "ERROR",
                            )
                            wcs_object = None

                    if wcs_object and wcs_object.is_celestial:
                        wcs_object.pixel_shape = (img_shape_hw_for_wcs[1], img_shape_hw_for_wcs[0])
                        try:
                            wcs_object._naxis1 = img_shape_hw_for_wcs[1]
                            wcs_object._naxis2 = img_shape_hw_for_wcs[0]
                        except AttributeError:
                            pass
                        if update_header_with_solution and fits_header is not None:
                            self._update_fits_header_with_wcs(
                                fits_header, wcs_object, solver_name="ASTAP"
                            )
                    else:
                        self._log(
                            "ASTAP: Échec création objet WCS ou WCS non céleste.",
                            "ERROR",
                        )
                        wcs_object = None
                else:
                    if is_boring_stack_disk_mode:
                        self._log(
                            "[ASTAP WCS] Sidecar .wcs missing, attempting FITS header fallback (batch_size=1 path)",
                            "DEBUG",
                        )
                        try:
                            with fits.open(image_path, memmap=False) as hdul:
                                hdr_fits = hdul[0].header.copy()
                            for card in hdr_fits.cards:
                                if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                                    card.value = str(card.value)
                            sanitize_header_for_wcs(hdr_fits)
                            wcs_object = WCS(hdr_fits, naxis=2, relax=True)
                            assert wcs_object.is_celestial
                            wcs_object.pixel_shape = (
                                img_shape_hw_for_wcs[1], img_shape_hw_for_wcs[0]
                            )
                            try:
                                wcs_object._naxis1 = img_shape_hw_for_wcs[1]
                                wcs_object._naxis2 = img_shape_hw_for_wcs[0]
                            except AttributeError:
                                pass
                            if update_header_with_solution and fits_header is not None:
                                self._update_fits_header_with_wcs(
                                    fits_header, wcs_object, solver_name="ASTAP"
                                )
                        except Exception as e_hdr_only:
                            self._log(
                                f"WCS parse failed from FITS header fallback: {e_hdr_only}",
                                "ERROR",
                            )
                            wcs_object = None
                    else:
                        self._log(
                            "ASTAP: Code retour 0 mais .wcs manquant/vide. Échec.",
                            "ERROR",
                        )
                        wcs_object = None
            else:
                log_msg_echec = f"ASTAP: Résolution échouée (code {result.returncode}"
                if not os.path.exists(expected_wcs_file): log_msg_echec += ", fichier .wcs NON trouvé"
                elif os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) == 0: log_msg_echec += ", fichier .wcs vide"
                else: log_msg_echec += ", .wcs trouvé mais autre problème possible"

                if os.path.exists(astap_log_file_generated):
                    try:
                        with open(astap_log_file_generated, "r", errors='ignore') as f_log_astap:
                            astap_log_content = f_log_astap.read(1000) # Lire un extrait
                        log_msg_echec += f". Extrait ASTAP Log: ...{astap_log_content[-400:]}" # Afficher la fin
                    except Exception as e_log_read:
                        log_msg_echec += f". (Erreur lecture log ASTAP: {e_log_read})"
                log_msg_echec += ")."
                self._log(log_msg_echec, "WARN")
                wcs_object = None

        except subprocess.TimeoutExpired:
            self._log(f"ASTAP: Timeout ({timeout_sec}s) expiré.", "ERROR")
            wcs_object = None
        except FileNotFoundError:
            self._log(f"ASTAP: Exécutable '{astap_exe_path}' non trouvé.", "ERROR")
            wcs_object = None
        except Exception as e:
            self._log(f"ASTAP: Erreur inattendue: {e}", "ERROR")
            traceback.print_exc(limit=1)
            wcs_object = None
        finally:
            # --- NETTOYAGE POST-EXÉCUTION ---
            self._log(f"ASTAP: Nettoyage post-exécution des fichiers temporaires...", "DEBUG")
            for f_to_clean_post in files_to_cleanup:
                if os.path.exists(f_to_clean_post):
                    try:
                        os.remove(f_to_clean_post)
                        self._log(f"ASTAP: Fichier '{os.path.basename(f_to_clean_post)}' nettoyé.", "DEBUG")
                    except Exception as e_del_post:
                        self._log(f"ASTAP: Avertissement - Échec nettoyage de '{os.path.basename(f_to_clean_post)}': {e_del_post}", "WARN")
            # --- FIN NETTOYAGE POST-EXÉCUTION ---

        return wcs_object


    def _parse_wcs_file_content(self, wcs_file_path, image_shape_hw):
        """Parse a ``.wcs`` file and return a :class:`~astropy.wcs.WCS` object."""

        if not os.path.exists(wcs_file_path) or os.path.getsize(wcs_file_path) == 0:
            self._log(f"Fichier WCS '{wcs_file_path}' non trouvé ou vide.", "ERROR")
            return None

        with open(wcs_file_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        clean, modified, dropped = _sanitize_astap_wcs_text(txt)
        if modified or dropped:
            self._log(
                f"Sanitised ASTAP WCS: modified={modified}, dropped={dropped}",
                "DEBUG",
            )

        hdr = fits.Header.fromstring(clean, sep="\n")

        for k, v in list(hdr.items()):
            if k == "CONTINUE":
                hdr[k] = str(v)
        while "HISTORY" in hdr:
            del hdr["HISTORY"]
        while "COMMENT" in hdr:
            del hdr["COMMENT"]

        _strip_redundant_scale_keywords(hdr)

        try:
            wcs_obj = WCS(hdr, naxis=2, relax=True, fix=True)
        except VerifyError:
            while "CONTINUE" in hdr:
                del hdr["CONTINUE"]
            wcs_obj = WCS(hdr, naxis=2, relax=True, fix=True)

        wcs_obj = _canonicalize_wcs_scale(wcs_obj)

        if wcs_obj is not None:
            try:
                wcs_obj.pixel_shape = (
                    int(image_shape_hw[1]),
                    int(image_shape_hw[0]),
                )
            except Exception:
                pass

        return wcs_obj

    def _update_fits_header_with_wcs(self, fits_header, wcs_object, solver_name="UnknownSolver"):
        """
        Met à jour un header FITS existant avec les informations d'un objet WCS.
        """
        if not fits_header or not wcs_object or not wcs_object.is_celestial:
            self._log("Mise à jour header annulée: header ou WCS invalide.", "WARN")
            return

        self._log(f"Mise à jour du header FITS avec la solution WCS de {solver_name}...", "DEBUG")
        try:
            # Effacer les anciennes clés WCS pour éviter les conflits, si elles existent.
            # C'est important car `fits_header.update(wcs_object.to_header())` peut ne pas
            # supprimer les anciennes clés si elles ne sont pas dans le nouveau header WCS.
            wcs_keys_to_remove = wcs_object.to_header(relax=True).keys() # Obtenir toutes les clés que WCS pourrait écrire
            # Ajouter d'autres clés WCS communes au cas où
            common_wcs_keys = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                               'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 
                               'CUNIT1', 'CUNIT2', 'CDELT1', 'CDELT2', 'CROTA2', 'EQUINOX', 'RADESYS',
                               'PV1_0', 'PV1_1', 'PV1_2', 'PV2_0', 'PV2_1', 'PV2_2'] # etc.
            for key_to_del in list(set(list(wcs_keys_to_remove) + common_wcs_keys)):
                if key_to_del in fits_header:
                    try:
                        del fits_header[key_to_del]
                    except KeyError: 
                        pass
            
            # Mettre à jour le header avec le nouveau WCS
            fits_header.update(wcs_object.to_header(relax=True)) 

            # Ajouter des informations sur la solution
            fits_header[f'{solver_name.upper()}_SOLVED'] = (True, f'{solver_name} solution found')
            if wcs_object.pixel_scale_matrix is not None:
                try:
                    pixscale_deg = np.sqrt(np.abs(np.linalg.det(wcs_object.pixel_scale_matrix)))
                    fits_header[f'{solver_name.upper()}_SCALE_ASEC'] = (
                        pixscale_deg * 3600.0, f'[arcsec/pix] Field scale from {solver_name}'
                    )
                except Exception: pass
            self._log("Header FITS mis à jour avec succès.", "DEBUG")
        except Exception as e_hdr_update:
            self._log(f"Erreur lors de la mise à jour du header FITS avec WCS: {e_hdr_update}", "ERROR")
            traceback.print_exc(limit=1)


def solve_image_wcs(
    image_path,
    fits_header,
    settings,
    update_header_with_solution=True,
    is_boring_stack_disk_mode=False,
    *,
    batch_size=None,
    final_combine=None,
):
    """Convenience wrapper for :class:`AstrometrySolver`.


    Parameters
    ----------
    image_path : str
        Path to the FITS image to solve.
    fits_header : astropy.io.fits.Header
        FITS header associated with the image (may be ``None``).
    settings : dict
        Dictionary of solver settings taken from :class:`SettingsManager`.
    update_header_with_solution : bool, optional
        If ``True`` the provided ``fits_header`` is updated with the solved WCS.
    is_boring_stack_disk_mode : bool, optional
        ``True`` only when running the disk-based pipeline with ``batch_size=1``.

    Returns
    -------
    astropy.wcs.WCS or None
        The solved WCS object, or ``None`` if solving failed.
    """
    try:
        solver = AstrometrySolver()
        return solver.solve(
            image_path,
            fits_header,
            settings,
            update_header_with_solution,
            is_boring_stack_disk_mode=is_boring_stack_disk_mode,
            batch_size=batch_size,
            final_combine=final_combine,
        )
    except Exception:
        return None


# --- END OF FILE seestar/alignment/astrometry_solver.py ---
