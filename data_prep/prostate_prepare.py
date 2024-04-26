import os
import shutil
import SimpleITK as sitk
import numpy as np


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  #
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  #
    resampler.SetReferenceImage(itkimage)  #
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  #
    return itkimgResampled


if __name__ == "__main__":
    print("start ...")
    groups = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]

    datapath = "/home/zhuzhanshi/download/CL_origin/continual prostate"
    data_out = "/Share8/zhuzhanshi/TED_DCMIS/storage/data"
    for subset in groups:
        source_path = os.path.join(datapath, subset)
        target_path = os.path.join(data_out, subset)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        for name in os.listdir(source_path):
            image = sitk.ReadImage(os.path.join(source_path, name))
            image_array = sitk.GetArrayFromImage(image)
            if "segmentation" in name:
                name = name.replace("segmentation", "gt")
                image_array[image_array > 0.0] = 1
            elif "Segmentation" in name:
                image_array[image_array > 0.0] = 1
                name = name.replace("Segmentation", "gt")
            else:
                image_array = (image_array - np.min(image_array)) / (
                    np.max(image_array) - np.min(image_array)
                )
                image_array = np.clip(image_array, 0.001, 0.99)
            image_new = sitk.GetImageFromArray(image_array)
            image_new.SetOrigin(image.GetOrigin())
            image_new.SetSpacing(image.GetSpacing())
            image_new.SetDirection(image.GetDirection())
            # print(image_array.shape)
            if "gt" in name:
                image_new = resize_image_itk(
                    image_new, (192, 192, image_array.shape[0]), 1
                )
            else:
                image_new = resize_image_itk(
                    image_new, (192, 192, image_array.shape[0]), 2
                )
            image_array = sitk.GetArrayFromImage(image_new)
            # print(image_array.shape)
            sitk.WriteImage(image_new, os.path.join(target_path, name))
        print(subset, "done !")
