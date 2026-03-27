#[cfg(test)]
mod tests_expanded {
      use super::*;
      use kornia_image::{Image, ImageSize, ImageError};
      use kornia_tensor::CpuAllocator;

    #[test]
      fn test_imgproc_extreme_sizes() -> Result<(), ImageError> {
                let size = ImageSize { width: 1, height: 1 };
                let img = Image::new(size, vec![100.0], CpuAllocator)?;
                let mut dst = Image::from_size_val(size, 0.0, CpuAllocator)?;

          box_blur(&img, &mut dst, (3, 3))?;
                assert_eq!(dst.as_slice()[0], 100.0);

          let size_rect = ImageSize { width: 1, height: 10 };
                let img_rect = Image::new(size_rect, vec![1.0; 10], CpuAllocator)?;
                let mut dst_rect = Image::from_size_val(size_rect, 0.0, CpuAllocator)?;
                gaussian_blur(&img_rect, &mut dst_rect, (3, 3), (0.5, 0.5))?;
                assert_eq!(dst_rect.numel(), 10);

          Ok(())
      }

    #[test]
      fn test_gaussian_blur_sigma_zero() -> Result<(), ImageError> {
                let size = ImageSize { width: 5, height: 5 };
                let img = Image::new(size, vec![1.0; 25], CpuAllocator)?;
                let mut dst = Image::from_size_val(size, 0.0, CpuAllocator)?;

          gaussian_blur(&img, &mut dst, (3, 3), (0.0, 0.0))?;
                for &val in dst.as_slice() {
                              assert!((val - 1.0).abs() < 1e-5);
                }
                Ok(())
      }

    #[test]
      fn test_sobel_mismatched_size_error() -> Result<(), ImageError> {
                let size = ImageSize { width: 5, height: 5 };
                let img = Image::new(size, vec![1.0; 25], CpuAllocator)?;
                let size_wrong = ImageSize { width: 4, height: 4 };
                let mut dst_wrong = Image::from_size_val(size_wrong, 0.0, CpuAllocator)?;

          let result = sobel::<1, _, _>(&img, &mut dst_wrong, 3);
                assert!(result.is_err());
                Ok(())
      }
}
