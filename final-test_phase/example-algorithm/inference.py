"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy
from PIL import Image
import numpy as np
import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("hazy-cardiac-ultrasound",): interface_0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interface_0_handler():

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    input_dir = INPUT_PATH / "images/hazy-cardiac-ultrasound"
    output_dir = OUTPUT_PATH / "images/dehazed-cardiac-ultrasound"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # For now, let us make bogus predictions
    #######################your own model##################

    image_array = load_image_file_as_array(location=input_dir)  # shape: [N, H, W, C]

    
    #######################################################
    # Save your output
    # Obtain a list of image file paths (used to retain original names when saving)
    image_paths = sorted(list(input_dir.glob("*.tiff")) +
                         list(input_dir.glob("*.tif")) +
                         list(input_dir.glob("*.mha")))
    for path, array in zip(image_paths, image_array):
        output_path = output_dir / path.name
        write_array_as_image_file(location=output_path, array=array)

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tiff")) +
        glob(str(location / "*.tif")) +
        glob(str(location / "*.mha"))
    )

    arrays = []
    for file_path in sorted(input_files):
        print(file_path)
        if file_path.endswith(".mha"):
            img = sitk.ReadImage(file_path)
            arr = sitk.GetArrayFromImage(img)  # [D, H, W] or [H, W]
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]  # middle slicing
        else:
            img = Image.open(file_path)
            arr = np.array(img)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)  # [H, W] -> [H, W, 1]
        arrays.append(arr)

    return arrays  # List[np.ndarray] 


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
