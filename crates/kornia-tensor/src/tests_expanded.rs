#[cfg(test)]
mod tests_expanded {
      use crate::allocator::CpuAllocator;
      use crate::tensor::{Tensor, TensorError};

      #[test]
      fn test_zero_sized_tensor() -> Result<(), TensorError> {
                let data: Vec<f32> = vec![];
                let t = Tensor::<f32, 2, _>::from_shape_vec([0, 10], data, CpuAllocator)?;
                assert_eq!(t.shape, [0, 10]);
                assert_eq!(t.numel(), 0);
                assert_eq!(t.strides, [10, 1]);
                Ok(())
            }

      #[test]
      fn test_single_element_tensor() -> Result<(), TensorError> {
                let data: Vec<f32> = vec![42.0];
                let t = Tensor::<f32, 1, _>::from_shape_vec([1], data, CpuAllocator)?;
                assert_eq!(t.shape, [1]);
                assert_eq!(t.numel(), 1);
                assert_eq!(t.get([0]), Some(&42.0));
                Ok(())
            }

      #[test]
      fn test_non_contiguous_view_reshape() -> Result<(), TensorError> {
                let data: Vec<u8> = (0..12).collect();
                let t = Tensor::<u8, 2, _>::from_shape_vec([3, 4], data, CpuAllocator)?;

                let view = t.permute_axes([1, 0]);
                assert_eq!(view.shape, [4, 3]);
                assert!(!view.is_contiguous());

                let reshaped = view.as_contiguous().reshape([2, 6])?;
                assert_eq!(reshaped.shape, [2, 6]);
                assert_eq!(reshaped.numel(), 12);
                Ok(())
            }

      #[test]
      fn test_allocator_error_handling() {
                let large_shape = [1usize << 30, 1usize << 30]; 
                let result = Tensor::<u8, 2, _>::from_shape_vec(large_shape, vec![], CpuAllocator);
                assert!(result.is_err());
            }
  }
