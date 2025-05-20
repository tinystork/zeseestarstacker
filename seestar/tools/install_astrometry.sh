#!/bin/bash

# Couleurs pour affichage
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
NC="\033[0m" # No Color

clear
echo -e "${GREEN}Bienvenue / Welcome to the Astrometry.net Index Installer${NC}"
echo "Ce script vous aide à installer astrometry.net et les index adaptés à votre télescope."
echo "This script installs astrometry.net and index files suitable for your telescope."
echo ""

# Étape 1 : installation astrometry.net
echo -e "${YELLOW}🔧 Mise à jour / Updating and installing astrometry.net...${NC}"
sudo apt update && sudo apt install -y astrometry.net wget

# Étape 2 : choix utilisateur / user choice
echo ""
echo -e "${YELLOW}🌌 Choisissez votre instrument / Choose your instrument:${NC}"
echo " 1 - Seestar S50"
echo " 2 - SkyWatcher Evoguide 50 + 183"
echo " 3 - Newton 150/750 + 183"
echo " 4 - SkyWatcher 72ED + 294MC"
echo " 5 - Celestron C8 + 0.63x"
echo " 6 - Celestron C11 + 0.63x"
echo " 7 - Celestron C14 + réducteur"
echo " 8 - Tous / All (long download)"
echo ""
echo -n "Votre choix / Your choice [1-8]: "
read choix

DEST=~/astrometry/index
mkdir -p "$DEST"
cd "$DEST" || exit 1

download_indexes() {
  for i in "$@"; do
    wget -nc http://data.astrometry.net/$i/index-$i.fits
  done
}

echo ""
case "$choix" in
  1)
    echo "📥 Seestar S50: 4206–4219"
    download_indexes {4206..4219}
    ;;
  2)
    echo "📥 Evoguide 50: 4205–4216"
    download_indexes {4205..4216}
    ;;
  3)
    echo "📥 Newton 150/750 + 183: 4206–4219"
    download_indexes {4206..4219}
    ;;
  4)
    echo "📥 72ED + 294MC: 4205–4217"
    download_indexes {4205..4217}
    ;;
  5)
    echo "📥 C8 + 0.63x: 4305–4310"
    download_indexes {4305..4310}
    ;;
  6)
    echo "📥 C11 + 0.63x: 4305–4310"
    download_indexes {4305..4310}
    ;;
  7)
    echo "📥 C14 + réducteur: 4310–4313"
    download_indexes {4310..4313}
    ;;
  8)
    echo "📥 Tous les profils: téléchargement complet / Full download"
    download_indexes {4205..4219} {4305..4313}
    ;;
  *)
    echo "⛔ Choix invalide / Invalid choice"
    exit 1
    ;;
esac

# Fin
echo ""
echo -e "${GREEN}✅ Terminé / Done!${NC}"
echo "Les index ont été installés dans : $DEST"
echo ""
echo "Utilisez solve-field comme ceci / Use solve-field like this:"
echo ""
echo -e "${YELLOW}solve-field your_image.fits --scale-units arcsecperpix --scale-low 0.25 --scale-high 2.5 \\"
echo "--dir results --overwrite --no-plots --index-x $DEST${NC}"
