"""
Stacking functions module for Seestar.
Contains methods for image stacking and related operations.
"""
import os
import time
import numpy as np
import traceback
from astropy.io import fits
import cv2

class StackingManager:
	"""
	Manages stacking operations for Seestar.
	"""
	
	def __init__(self, main_window):
		"""
		Initialize the stacking manager.
		
		Args:
			main_window: The main SeestarStackerGUI instance
		"""
		self.main = main_window
	
	def run_enhanced_traditional_stacking(self, input_folder, output_folder):
		"""
		Execute an enhanced traditional stacking process with real-time visualization.
		This function keeps the basic alignment but adds visualization features
		and progressive stack combination.

		Args:
			input_folder (str): Folder containing raw images
			output_folder (str): Output folder for stacked images
		"""
		try:
			# Create necessary folders
			os.makedirs(output_folder, exist_ok=True)
			aligned_folder = os.path.join(output_folder, "aligned_temp")
			os.makedirs(aligned_folder, exist_ok=True)

			# Get input files
			all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
			if not all_files:
				self.main.update_progress("❌ No .fit/.fits files found")
				return

			# Estimate batch size if auto
			if self.main.batch_size.get() <= 0:
				from seestar.core.utils import estimate_batch_size
				sample_path = os.path.join(input_folder, all_files[0])
				self.main.batch_size.set(estimate_batch_size(sample_path))
				self.main.update_progress(f"🧠 Automatic batch size estimated: {self.main.batch_size.get()}")

			batch_size = self.main.batch_size.get()
			total_files = len(all_files)
			self.main.update_progress(f"🔍 {total_files} images found to process in batches of {batch_size}")
			
			# Initialize global image counter and start time
			self.main.total_images_count = total_files
			self.main.processed_images_count = 0
			self.main.global_start_time = time.time()
			
			# Step 1: Get a reference image
			# (use the aligner's method to select the reference)
			self.main.update_progress("🔍 Searching for reference image...")
			
			# If the user specified a reference image, use it
			if self.main.reference_image_path.get():
				reference_files = None  # The aligner will use reference_image_path directly
			else:
				# Otherwise use up to 50 images to find a good reference
				reference_files = all_files[:min(50, len(all_files))]
				
			self.main.aligner.stop_processing = False
			reference_folder = self.main.aligner.align_images(
				input_folder, 
				aligned_folder, 
				specific_files=reference_files
			)
			
			# Check that the reference image exists
			reference_image_path = os.path.join(reference_folder, "reference_image.fit")
			if not os.path.exists(reference_image_path):
				self.main.update_progress("❌ Failed to create reference image")
				return
			
			self.main.update_progress(f"⭐ Reference image created: {reference_image_path}")
			
			# Clean the temporary alignment folder
			for f in os.listdir(aligned_folder):
				if f != "reference_image.fit" and f != "unaligned":
					os.remove(os.path.join(aligned_folder, f))
			
			# Process images in batches
			start_time = time.time()
			stack_count = 0
			batch_count = (total_files + batch_size - 1) // batch_size  # Ceiling division
			
			for batch_idx in range(batch_count):
				if self.main.aligner.stop_processing or self.main.stacker.stop_processing:
					self.main.update_progress("⛔ Processing stopped by user.")
					break
					
				# Calculate indices of current batch
				batch_start = batch_idx * batch_size
				batch_end = min(batch_start + batch_size, total_files)
				current_files = all_files[batch_start:batch_end]
				
				self.main.update_progress(
					f"🚀 Processing batch {batch_idx + 1}/{batch_count} "
					f"(images {batch_start+1} to {batch_end}/{total_files})...",
					batch_idx * 100.0 / batch_count
				)
				
				# Step 2: Align batch images to reference image
				self.main.update_progress(f"📐 Aligning images for batch {batch_idx + 1}...")
				
				# Use the reference image as explicit reference for all batches
				original_ref_path = self.main.aligner.reference_image_path
				self.main.aligner.reference_image_path = reference_image_path
				
				# Align the batch images
				self.main.aligner.stop_processing = False
				aligned_result = self.main.aligner.align_images(
					input_folder, 
					aligned_folder, 
					specific_files=current_files
				)
				
				# Update counters for each image processed in this batch
				images_in_this_batch = len(current_files)
				self.main.processed_images_count += images_in_this_batch
				
				# Update time per image and global estimation
				elapsed_time = time.time() - self.main.global_start_time
				if self.main.processed_images_count > 0:
					self.main.time_per_image = elapsed_time / self.main.processed_images_count
					remaining_time = self.main.processing_manager.calculate_remaining_time()
					self.main.remaining_time_var.set(remaining_time)
				
				# Restore original reference
				self.main.aligner.reference_image_path = original_ref_path
				
				# Check if alignment was stopped
				if self.main.aligner.stop_processing:
					self.main.update_progress("⛔ Alignment stopped by user.")
					break
				
				# Step 3: Stack aligned images from the batch
				self.main.update_progress(f"🧮 Stacking aligned images from batch {batch_idx + 1}...")
				
				# Find aligned files (format aligned_XXXX.fit)
				aligned_files = [f for f in os.listdir(aligned_folder) 
							   if f.startswith('aligned_') and f.endswith('.fit')]
				
				if not aligned_files:
					self.main.update_progress(f"⚠️ No aligned images found for batch {batch_idx + 1}")
					continue
				
				# Stack aligned images from batch
				stack_file = os.path.join(output_folder, f"stack_batch_{batch_idx+1:03d}.fit")
				
				# Load and stack aligned images
				batch_stack_data, batch_stack_header = self._stack_aligned_images(
					aligned_folder, aligned_files, stack_file
				)
				
				if batch_stack_data is None:
					self.main.update_progress(f"⚠️ Failed to stack batch {batch_idx + 1}")
					continue
				
				stack_count += 1
				
				# Update preview with batch result
				self.main.update_preview(
					batch_stack_data, 
					f"Batch {batch_idx + 1} stack", 
					self.main.apply_stretch.get()
				)

				# Step 4: Combine with cumulative stack if not first batch
				if self.main.current_stack_data is None:
					# First batch, initialize cumulative stack
					self.main.current_stack_data = batch_stack_data
					self.main.current_stack_header = batch_stack_header
				else:
					# Combine batch stack with cumulative stack
					self.main.update_progress(f"🔄 Merging with cumulative stack...")
					self._combine_with_current_stack(batch_stack_data, batch_stack_header)

				# Create enhanced stack name with progress information
				progress_info = f" ({batch_idx + 1}/{batch_count} batches)"
				enhanced_stack_name = self.main.processing_manager.enhance_stack_name(
					f"Cumulative stack{progress_info}",
					self.main.total_images_count,
					self.main.processed_images_count,
					self.main.additional_folders,
					self.main.total_additional_counted,
					self.main.processing_additional,
					self.main.batch_size.get()
				)

				# Update preview with cumulative stack
				self.main.update_preview(
					self.main.current_stack_data, 
					enhanced_stack_name, 
					self.main.apply_stretch.get()
				)
				
				# Save cumulative stack
				cumulative_file = os.path.join(output_folder, f"stack_cumulative.fit")
				from seestar.core.image_processing import save_fits_image, save_preview_image
				save_fits_image(self.main.current_stack_data, cumulative_file, self.main.current_stack_header)
				save_preview_image(self.main.current_stack_data, os.path.join(output_folder, "stack_cumulative.png"))
				
				# Save a version with original metadata
				if self.main.current_stack_data is not None and all_files:
					ref_img_path = os.path.join(input_folder, all_files[0])  # Use first image as reference
					color_output = os.path.join(output_folder, "stack_final_color_metadata.fit")
					self.save_stack_with_original_metadata(self.main.current_stack_data, color_output, ref_img_path)
				
				# Delete processed aligned images if requested
				if self.main.remove_aligned.get():
					for f in aligned_files:
						try:
							os.remove(os.path.join(aligned_folder, f))
						except Exception as e:
							self.main.update_progress(f"⚠️ Cannot delete {f}: {e}")
					
					self.main.update_progress(f"🧹 Aligned images from batch {batch_idx + 1} deleted")
			
			# Apply denoising to final stack if requested
			if self.main.apply_denoise.get() and self.main.current_stack_data is not None:
				self.main.update_progress("🧹 Applying denoising to final stack...")
				try:
					from seestar.core.utils import apply_denoise
					self.main.current_stack_data = apply_denoise(self.main.current_stack_data, strength=5)
					
					# Update header
					self.main.current_stack_header['DENOISED'] = True
					
					# Save denoised version
					final_file = os.path.join(output_folder, "stack_final_denoised.fit")
					from seestar.core.image_processing import save_fits_image, save_preview_image
					save_fits_image(self.main.current_stack_data, final_file, self.main.current_stack_header)
					save_preview_image(self.main.current_stack_data, os.path.join(output_folder, "stack_final_denoised.png"))
					
					# Update preview
					self.main.update_preview(
						self.main.current_stack_data, 
						"Final denoised stack", 
						self.main.apply_stretch.get()
					)
				except Exception as e:
					self.main.update_progress(f"⚠️ Denoising failed: {e}")
			
			# Final report
			if stack_count > 0:
				self.main.update_progress(f"✅ Processing completed: {stack_count} batches stacked")
			else:
				self.main.update_progress("⚠️ No stacks were created")

			# Clean unaligned files
			self.cleanup_unaligned_files(output_folder)
			
		except Exception as e:
			self.main.update_progress(f"❌ Error during processing: {e}")
			traceback.print_exc()
	
	def process_additional_folders(self, output_folder):
		"""
		Process all additional folders that were added during processing.
		Images are processed in batches in the same way as the main folder.
		
		Args:
			output_folder (str): Output folder for results
		"""
		if not self.main.additional_folders:
			return
			
		self.main.processing_additional = True
		self.main.update_progress(f"📂 Processing {len(self.main.additional_folders)} additional folders...")
		
		# Get reference image path
		reference_image_path = os.path.join(output_folder, "aligned_temp", "reference_image.fit")
		
		if not os.path.exists(reference_image_path):
			self.main.update_progress("❌ Reference image not found. Cannot process additional folders.")
			return
			
		self.main.update_progress(f"⭐ Using existing reference image: {reference_image_path}")
		
		# For each additional folder
		for folder_idx, folder in enumerate(self.main.additional_folders):
			if not self.main.processing:
				self.main.update_progress("⛔ Additional folder processing stopped.")
				break
				
			self.main.update_progress(f"📂 Processing additional folder {folder_idx+1}/{len(self.main.additional_folders)}: {folder}")
			
			# Check if folder exists and contains FITS files
			all_files = [f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))]
			
			if not all_files:
				self.main.update_progress(f"⚠️ Folder {folder} contains no FITS files. Skipped.")
				continue
			
			# Create a temporary folder for aligned images from this folder
			aligned_folder = os.path.join(output_folder, f"aligned_additional_{folder_idx+1}")
			os.makedirs(aligned_folder, exist_ok=True)
			
			# Use same batch size as for main processing
			batch_size = self.main.batch_size.get()
			if batch_size <= 0:
				batch_size = 10  # Default value if not defined
			
			total_files = len(all_files)
			self.main.update_progress(f"🔍 {total_files} images found to process in batches of {batch_size}")
			
			# Process in batches
			batch_count = (total_files + batch_size - 1) // batch_size  # Ceiling division
			start_time = time.time()
			
			for batch_idx in range(batch_count):
				if not self.main.processing:
					self.main.update_progress("⛔ Processing stopped by user.")
					break
					
				# Calculate indices of current batch
				batch_start = batch_idx * batch_size
				batch_end = min(batch_start + batch_size, total_files)
				current_files = all_files[batch_start:batch_end]
				
				self.main.update_progress(
					f"🚀 Processing batch {batch_idx + 1}/{batch_count} "
					f"(images {batch_start+1} to {batch_end}/{total_files})...",
					(batch_idx * 100.0) / batch_count
				)
				
				# Configure aligner to use existing reference image
				self.main.aligner.stop_processing = False
				original_ref_path = self.main.aligner.reference_image_path
				self.main.aligner.reference_image_path = reference_image_path
				
				# Align batch images
				self.main.aligner.align_images(
					folder, 
					aligned_folder, 
					specific_files=current_files
				)
				
				# Restore original reference
				self.main.aligner.reference_image_path = original_ref_path
				
				# Check if alignment was stopped
				if self.main.aligner.stop_processing:
					self.main.update_progress("⛔ Alignment stopped by user.")
					break
					
				# Find aligned files (format aligned_XXXX.fit)
				aligned_files = [f for f in os.listdir(aligned_folder) 
							   if f.startswith('aligned_') and f.endswith('.fit')]
				
				if not aligned_files:
					self.main.update_progress(f"⚠️ No aligned images found for batch {batch_idx + 1}")
					continue
					
				# Stack aligned images from batch
				self.main.update_progress(f"🧮 Stacking aligned images from batch {batch_idx + 1}...")
				
				# Create a temporary stack for this batch
				batch_stack_file = os.path.join(output_folder, f"stack_additional_{folder_idx+1}_batch_{batch_idx+1}.fit")
				
				# Stack aligned images from this batch
				batch_stack_data, batch_stack_header = self._stack_aligned_images(
					aligned_folder, aligned_files, batch_stack_file
				)
				
				if batch_stack_data is None:
					self.main.update_progress(f"⚠️ Failed to stack batch {batch_idx + 1}")
					continue
					
				# Update preview with batch result
				self.main.update_preview(
					batch_stack_data, 
					f"Batch {batch_idx + 1} stack (folder {folder_idx+1})", 
					self.main.apply_stretch.get()
				)
				
				# Combine with cumulative stack
				if self.main.current_stack_data is None:
					# First stack, initialize cumulative stack
					self.main.current_stack_data = batch_stack_data
					self.main.current_stack_header = batch_stack_header
				else:
					# Combine with existing stack
					self.main.update_progress("🔄 Combining with cumulative stack...")
					self._combine_with_current_stack(batch_stack_data, batch_stack_header)
				
				# Update preview with cumulative stack
				self.main.update_preview(
					self.main.current_stack_data, 
					f"Cumulative stack (after folder {folder_idx+1}, batch {batch_idx+1}/{batch_count})", 
					self.main.apply_stretch.get()
				)
				
				# Save cumulative stack
				cumulative_file = os.path.join(output_folder, "stack_cumulative.fit")
				from seestar.core.image_processing import save_fits_image, save_preview_image
				save_fits_image(self.main.current_stack_data, cumulative_file, self.main.current_stack_header)
				save_preview_image(self.main.current_stack_data, os.path.join(output_folder, "stack_cumulative.png"))
				
				# Final stack with metadata
				color_output = os.path.join(output_folder, "stack_final_color_metadata.fit")
				self.save_stack_with_original_metadata(self.main.current_stack_data, color_output, 
													os.path.join(folder, all_files[0]))
				
				# Delete processed aligned images if requested
				if self.main.remove_aligned.get():
					for f in aligned_files:
						try:
							os.remove(os.path.join(aligned_folder, f))
						except Exception as e:
							self.main.update_progress(f"⚠️ Cannot delete {f}: {e}")
					
					self.main.update_progress(f"🧹 Aligned images from batch {batch_idx + 1} deleted")
			
			# Report for this folder
			self.main.update_progress(f"✅ Folder {folder_idx+1}/{len(self.main.additional_folders)} processed successfully")
		
		# Apply denoising to final stack if requested
		if self.main.apply_denoise.get() and self.main.current_stack_data is not None:
			self.main.update_progress("🧹 Applying denoising to final stack...")
			try:
				from seestar.core.utils import apply_denoise
				self.main.current_stack_data = apply_denoise(self.main.current_stack_data, strength=5)
				
				# Update header
				self.main.current_stack_header['DENOISED'] = True
				
				# Save denoised version
				final_file = os.path.join(output_folder, "stack_final_denoised.fit")
				from seestar.core.image_processing import save_fits_image, save_preview_image
				save_fits_image(self.main.current_stack_data, final_file, self.main.current_stack_header)
				save_preview_image(self.main.current_stack_data, os.path.join(output_folder, "stack_final_denoised.png"))
				
				# Update preview
				self.main.update_preview(
					self.main.current_stack_data, 
					"Final denoised stack", 
					self.main.apply_stretch.get()
				)
			except Exception as e:
				self.main.update_progress(f"⚠️ Denoising failed: {e}")
		
		self.main.processing_additional = False
		self.main.update_progress("🏁 Additional folder processing completed.")
		
		# Clean unaligned files
		self.cleanup_unaligned_files(output_folder)
	
	def _stack_aligned_images(self, aligned_folder, aligned_files, output_file):
		"""
		Stack aligned images and save the result.
		
		Args:
			aligned_folder (str): Folder containing aligned images
			aligned_files (list): List of files to stack
			output_file (str): Output file path
			
		Returns:
			tuple: (stack_data, stack_header) or (None, None) if failed
		"""
		try:
			from seestar.core.image_processing import load_and_validate_fits, save_fits_image, save_preview_image
			
			# Load all images from batch
			images = []
			headers = []
			
			for file in aligned_files:
				try:
					file_path = os.path.join(aligned_folder, file)
					img = load_and_validate_fits(file_path)
					
					# If image is 3D with first dimension as channels (3xHxW), 
					# convert to HxWx3
					if img.ndim == 3 and img.shape[0] == 3:
						img = np.moveaxis(img, 0, -1)
					
					images.append(img)
					headers.append(fits.getheader(file_path))
				except Exception as e:
					self.main.update_progress(f"⚠️ Error loading {file}: {e}")
			
			if not images:
				self.main.update_progress(f"❌ No valid images found to stack")
				return None, None
			
			# Stack images according to chosen method
			stacking_mode = self.main.stacking_mode.get()
			self.main.update_progress(f"🧮 Stacking with '{stacking_mode}' method...")
			
			if stacking_mode == "mean":
				stacked_image = np.mean(images, axis=0)
			elif stacking_mode == "median":
				stacked_image = np.median(np.stack(images, axis=0), axis=0)
			elif stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
				kappa = self.main.kappa.get()
				# Convert image list to 3D/4D array
				stack = np.stack(images, axis=0)
				
				# Calculate mean and standard deviation
				mean = np.mean(stack, axis=0)
				std = np.std(stack, axis=0)
				
				if stacking_mode == "kappa-sigma":
					# For kappa-sigma, create masks for each image
					sum_image = np.zeros_like(mean)
					mask_sum = np.zeros_like(mean)
					
					for img in stack:
						deviation = np.abs(img - mean)
						mask = deviation <= (kappa * std)
						sum_image += img * mask
						mask_sum += mask
					
					# Avoid division by zero
					mask_sum = np.maximum(mask_sum, 1)
					stacked_image = sum_image / mask_sum
					
				elif stacking_mode == "winsorized-sigma":
					# For winsorized-sigma, replace extreme values
					upper_bound = mean + kappa * std
					lower_bound = mean - kappa * std
					
					# Apply limits to each image
					clipped_stack = np.clip(stack, lower_bound, upper_bound)
					
					# Calculate mean of clipped images
					stacked_image = np.mean(clipped_stack, axis=0)
			else:
				# Fallback to mean if method not recognized
				self.main.update_progress(f"⚠️ Stacking method '{stacking_mode}' not recognized, using 'mean'")
				stacked_image = np.mean(images, axis=0)
			
			# Create FITS header
			stack_header = fits.Header()
			stack_header['STACKED'] = True
			stack_header['STACKTYP'] = stacking_mode
			stack_header['NIMAGES'] = len(images)
			
			if stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
				stack_header['KAPPA'] = self.main.kappa.get()
			
			# Preserve important metadata from first image
			important_keys = ['INSTRUME', 'EXPTIME', 'FILTER', 'OBJECT', 'DATE-OBS']
			for key in important_keys:
				if headers and key in headers[0]:
					stack_header[key] = headers[0][key]
			
			# Save stack
			save_fits_image(stacked_image, output_file, header=stack_header)
			
			# Save PNG preview
			preview_file = os.path.splitext(output_file)[0] + ".png"
			save_preview_image(stacked_image, preview_file)
			
			self.main.update_progress(f"✅ Stack created and saved: {output_file}")
			
			return stacked_image, stack_header
			
		except Exception as e:
			self.main.update_progress(f"❌ Error during stacking: {e}")
			traceback.print_exc()
			return None, None
	
def _combine_with_current_stack(self, new_stack_data, new_stack_header):
	"""
	Combine new stack with current cumulative stack.
	
	Args:
		new_stack_data (numpy.ndarray): New stack data
		new_stack_header (astropy.io.fits.Header): New stack header
	"""
	try:
		# Check that dimensions are compatible
		if new_stack_data.shape != self.main.current_stack_data.shape:
			self.main.update_progress(f"⚠️ Incompatible dimensions: current stack {self.main.current_stack_data.shape}, new {new_stack_data.shape}")
			
			# Try resizing (implement if needed)
			return
		
		# Get number of images in each stack
		current_images = int(self.main.current_stack_header.get('NIMAGES', 1))
		new_images = int(new_stack_header.get('NIMAGES', 1))
		total_images = current_images + new_images
		
		# Check that same stacking method is used
		current_method = self.main.current_stack_header.get('STACKTYP', 'mean')
		new_method = new_stack_header.get('STACKTYP', 'mean')
		
		if current_method != new_method:
			self.main.update_progress(f"⚠️ Different stacking methods: current {current_method}, new {new_method}")
			# Continue anyway, but note the difference
		
		# Combine with weighting by number of images
		weight_current = current_images / total_images
		weight_new = new_images / total_images
		
		self.main.current_stack_data = (self.main.current_stack_data * weight_current) + (new_stack_data * weight_new)
		
		# Update header
		self.main.current_stack_header['NIMAGES'] = total_images
		self.main.current_stack_header['STACKTYP'] = current_method  # Keep original method
		
		# Update preview with new combined stack data
		# Create meaningful stack name that includes progress information
		enhanced_stack_name = self.main.processing_manager.enhance_stack_name(
			"Cumulative stack",
			self.main.total_images_count,
			self.main.processed_images_count,
			self.main.additional_folders,
			self.main.total_additional_counted,
			self.main.processing_additional,
			self.main.batch_size.get()
		)
		
		# Update the preview through the preview manager
		self.main.update_preview(
			stack_data=self.main.current_stack_data,
			stack_name=enhanced_stack_name,
			apply_stretch=self.main.apply_stretch.get()
		)
		
	except Exception as e:
		self.main.update_progress(f"❌ Error combining stacks: {e}")
		import traceback
		traceback.print_exc()

	
	def cleanup_unaligned_files(self, output_folder):
		"""
		Clean unaligned files in temporary folders.
		
		Args:
			output_folder (str): Output folder containing temporary subfolders
		"""
		self.main.update_progress("🧹 Cleaning unaligned files...")
		
		try:
			# Clean unaligned folder from main processing
			main_unaligned_folder = os.path.join(output_folder, "aligned_temp", "unaligned")
			if os.path.exists(main_unaligned_folder):
				file_count = 0
				for file in os.listdir(main_unaligned_folder):
					try:
						os.remove(os.path.join(main_unaligned_folder, file))
						file_count += 1
					except Exception as e:
						self.main.update_progress(f"⚠️ Cannot delete {file}: {e}")
				
				if file_count > 0:
					self.main.update_progress(f"🧹 {file_count} unaligned files deleted from main folder")
				
				# Delete folder if empty
				try:
					os.rmdir(main_unaligned_folder)
					self.main.update_progress("🧹 Main 'unaligned' folder deleted")
				except:
					# If folder is not empty, that's not a problem
					pass
			
			# Clean unaligned folders from additional processing
			for folder_idx in range(len(self.main.additional_folders)):
				add_unaligned_folder = os.path.join(output_folder, f"aligned_additional_{folder_idx+1}", "unaligned")
				
				if os.path.exists(add_unaligned_folder):
					file_count = 0
					for file in os.listdir(add_unaligned_folder):
						try:
							os.remove(os.path.join(add_unaligned_folder, file))
							file_count += 1
						except Exception as e:
							self.main.update_progress(f"⚠️ Cannot delete {file}: {e}")
					
					if file_count > 0:
						self.main.update_progress(f"🧹 {file_count} unaligned files deleted from additional folder {folder_idx+1}")
					
					# Delete folder if empty
					try:
						os.rmdir(add_unaligned_folder)
						self.main.update_progress(f"🧹  'unaligned' folder for additional folder {folder_idx+1} deleted")
					except:
						# If folder is not empty, that's not a problem
						pass
			
			self.main.update_progress("✅ Unaligned files cleanup completed")
		
		except Exception as e:
			self.main.update_progress(f"⚠️ Error during unaligned files cleanup: {e}")
	
	def save_stack_with_original_metadata(self, stacked_image, output_path, original_path=None):
		"""
		Save a color stacked image with metadata from original image.
		
		Args:
			stacked_image (numpy.ndarray): RGB stacked image (HxWx3)
			output_path (str): Output file path
			original_path (str): Path to an original image to get metadata from
		"""
		# Create a basic header
		new_header = fits.Header()
		
		# If image is RGB (HxWx3), convert to 3xHxW format for FITS
		if stacked_image.ndim == 3 and stacked_image.shape[2] == 3:
			# Convert from HxWx3 to 3xHxW
			fits_image = np.moveaxis(stacked_image, -1, 0)
			new_header['NAXIS'] = 3
			new_header['NAXIS1'] = stacked_image.shape[1]  # Width
			new_header['NAXIS2'] = stacked_image.shape[0]  # Height
			new_header['NAXIS3'] = 3  # 3 channels
			new_header['CTYPE3'] = 'RGB'
		else:
			fits_image = stacked_image
			new_header['NAXIS'] = 2
			new_header['NAXIS1'] = stacked_image.shape[1]
			new_header['NAXIS2'] = stacked_image.shape[0]
		
		new_header['BITPIX'] = 16
		
		# If an original image is provided, get its metadata
		if original_path and os.path.exists(original_path):
			try:
				orig_header = fits.getheader(original_path)
				
				# List of keys to preserve from original
				keys_to_preserve = [
					'TELESCOP', 'INSTRUME', 'EXPTIME', 'FILTER',
					'RA', 'DEC', 'FOCALLEN', 'APERTURE', 'SITELONG', 'SITELAT',
					'CCD-TEMP', 'GAIN', 'XPIXSZ', 'YPIXSZ', 'FOCUSPOS',
					'OBJECT', 'DATE-OBS', 'CREATOR', 'PRODUCER', 'PROGRAM'
				]
				
				# Copy metadata from original
				for key in keys_to_preserve:
					if key in orig_header:
						new_header[key] = orig_header[key]
			except Exception as e:
				print(f"Error retrieving original metadata: {e}")
		
		# Add stacking information
		new_header['STACKED'] = True
		new_header['STACKTYP'] = 'color-stack'
		new_header['BAYERPAT'] = 'N/A'  # Indicate this is no longer a Bayer image
		
		# Save image
		fits_image = cv2.normalize(fits_image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
		fits.writeto(output_path, fits_image, new_header, overwrite=True)
		print(f"Color stacked image saved: {output_path}")