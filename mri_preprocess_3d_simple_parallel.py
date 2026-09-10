import sys
import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import random
from HD_BET.hd_bet import hd_bet
import argparse
import torch
import multiprocessing as mp
from functools import partial
import time


def _image_id_from_path(img_path):
    """Return scan ID from a .nii path."""
    name = os.path.basename(img_path)
    if name.endswith('.nii'):
        return name[:-4]
    return os.path.splitext(name)[0]


def _collect_nifti_files(input_dir):
    """Recursively collect .nii files, skipping masks."""
    input_files = sorted(glob.glob(os.path.join(input_dir, '**/*.nii'), recursive=True))
    # Exclude .nii.gz accidentally matched on some systems; only keep true .nii
    input_files = [f for f in input_files if f.endswith('.nii') and not f.endswith('.nii.gz')]
    return [f for f in input_files if '_mask' not in os.path.basename(f)]


def brain_extraction(input_dir, output_dir, device):
    """
    Brain extraction using HDBET package (UNet based DL method).

    Writes skull-stripped images to ``output_dir/images`` and masks to
    ``output_dir/masks``.

    Args:
        input_dir {path} -- input directory for registered images
        output_dir {path} -- cohort output root (images/ and masks/ created here)
        device {str} -- HD-BET device id or ``cpu``
    """
    import shutil

    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print("Running brain extraction...")
    print(f"Input directory: {input_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")

    # HD-BET writes both skull-stripped volumes and masks into images_dir
    hd_bet(input_dir, images_dir, device=device, mode='fast', tta=0, save_mask=1)

    # Move masks into the dedicated masks folder
    moved = 0
    for name in os.listdir(images_dir):
        if '_mask' not in name:
            continue
        src = os.path.join(images_dir, name)
        dst = os.path.join(masks_dir, name)
        shutil.move(src, dst)
        moved += 1

    print('Brain extraction complete!')
    print(f"Moved {moved} mask files to {masks_dir}")
    print(f"Skull-stripped images in {images_dir}: {len(os.listdir(images_dir))}")
    print(f"Masks in {masks_dir}: {len(os.listdir(masks_dir))}")

def process_single_image(args):
    """Process a single image registration"""
    img_file, temp_img_path, output_dir, interp_type = args
    
    try:
        # Read template and moving images
        fixed_img = sitk.ReadImage(temp_img_path, sitk.sitkFloat32)
        moving_img = sitk.ReadImage(img_file, sitk.sitkFloat32)
        
        # Get ID from filename
        ID = _image_id_from_path(img_file)
        
        # Apply bias field correction with correct parameters
        maskImage = sitk.OtsuThreshold(moving_img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([2, 1, 1])  # Reduced iterations for speed
        moving_img = corrector.Execute(moving_img, maskImage)

        # Set interpolation type
        if interp_type == 'linear':
            interp = sitk.sitkLinear
        elif interp_type == 'bspline':
            interp = sitk.sitkBSpline
        elif interp_type == 'nearest_neighbor':
            interp = sitk.sitkNearestNeighbor

        # Resample fixed image to 1mm isotropic
        old_size = fixed_img.GetSize()
        old_spacing = fixed_img.GetSpacing()
        new_spacing = (1, 1, 1)
        new_size = [
            int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
            int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
            int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
        ]

        # Resample fixed image
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputOrigin(fixed_img.GetOrigin())
        resample.SetOutputDirection(fixed_img.GetDirection())
        resample.SetInterpolator(interp)
        resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
        resample.SetOutputPixelType(sitk.sitkFloat32)
        fixed_img_resampled = resample.Execute(fixed_img)

        # Initialize transform
        transform = sitk.CenteredTransformInitializer(
            fixed_img_resampled, 
            moving_img, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Set up registration method with optimized parameters
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.5,
            numberOfIterations=50,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=5)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[3, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(transform)

        # Execute registration
        final_transform = registration_method.Execute(fixed_img_resampled, moving_img)

        # Apply transform and save registered image
        moving_img_resampled = sitk.Resample(
            moving_img, 
            fixed_img_resampled, 
            final_transform, 
            sitk.sitkLinear, 
            0.0, 
            moving_img.GetPixelID())
        
        # Save with _0000 suffix as required by HD-BET
        output_filename = os.path.join(output_dir, f"{ID}_0000.nii.gz")
        sitk.WriteImage(moving_img_resampled, output_filename)
        
        print(f"Successfully processed: {ID}")
        return True, ID, output_filename
        
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        return False, _image_id_from_path(img_file), None


def registration(input_dir, output_dir, temp_img, interp_type='linear', num_workers=None):
    """
    Parallel MRI registration with SimpleITK
    Args:
        input_dir {path} -- Directory containing input images (searched recursively for .nii)
        output_dir {path} -- Directory to save registered images
        temp_img {str} -- Registration image template
        num_workers {int|None} -- Worker processes; defaults to cpu_count - 2
    Returns:
        bool -- Success status
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursively find .nii under patient subfolders
    input_files = _collect_nifti_files(input_dir)
    if not input_files:
        print(f"No .nii files found under {input_dir}")
        return False
    
    if num_workers is None:
        num_processes = max(1, mp.cpu_count() - 2)
    else:
        num_processes = max(1, int(num_workers))
    print(f"Found {len(input_files)} images under {input_dir}")
    print(f"Using {num_processes} processes for parallel registration")
    
    # Prepare arguments for parallel processing
    process_args = [(f, temp_img, output_dir, interp_type) for f in input_files]
    
    # Initialize counters
    successful_registrations = 0
    total_processed = 0
    
    # Process images in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance
        results = list(tqdm(
            pool.imap_unordered(process_single_image, process_args),
            total=len(input_files),
            desc="Registering images"
        ))
        
        # Process results
        for success, ID, output_file in results:
            total_processed += 1
            if success:
                successful_registrations += 1
                print(f"Successfully registered: {ID}")
                print(f"Saved to: {output_file}")
    
    print(f"\nSuccessfully registered {successful_registrations} out of {total_processed} images.")
    print(f"Contents of output directory {output_dir}:")
    print(os.listdir(output_dir))
    
    return successful_registrations > 0

def main(temp_img, input_dir, output_dir, num_workers=None):
    """
    Main function to process brain MRI images
    Args:
        temp_img {str} -- Path to template image
        input_dir {str} -- Path to input directory containing images
        output_dir {str} -- Path to output directory for results
        num_workers {int|None} -- Worker processes for registration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # set device
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create temporary directory for intermediate results
    temp_reg_dir = os.path.join(output_dir, 'temp_registered')
    os.makedirs(temp_reg_dir, exist_ok=True)
    
    print("Starting brain MRI preprocessing...")
    
    # Registration
    print("\nStep 1: Image Registration")
    success = registration(
        input_dir=input_dir,
        output_dir=temp_reg_dir,
        temp_img=temp_img,
        num_workers=num_workers,
    )
    
    if not success:
        print("Registration failed! No images were processed successfully.")
        return
    
    print("\nChecking temporary directory contents:")
    print(os.listdir(temp_reg_dir))
    
    # Skull stripping
    print("\nStep 2: Brain Extraction")
    brain_extraction(
        input_dir=temp_reg_dir,
        output_dir=output_dir,
        device=device
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_reg_dir)
    
    print("\nPreprocessing complete!")
    print(f"Skull-stripped images: {os.path.join(output_dir, 'images')}")
    print(f"Masks: {os.path.join(output_dir, 'masks')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process brain MRI registration and skull stripping.")
    parser.add_argument("--temp_img", type=str, required=True, help="Path to the atlas template image.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed images.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU worker processes for registration "
             "(default: cpu_count - 2). Recommended ~24-32 on large hosts.",
    )

    args = parser.parse_args()
    time_start = time.time()
    main(
        temp_img=args.temp_img,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    time_end = time.time()
    print(f"Total processing time: {time_end - time_start:.2f} seconds")