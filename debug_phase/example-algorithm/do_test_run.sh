#!/usr/bin/env bash
set -ex
# Stop at first error
#set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example-algorithm-timeline"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"


if [ -d "${OUTPUT_DIR}/interface_0" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "${OUTPUT_DIR}/interface_0"

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interface_0":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_0"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    ## Note the extra arguments that are passed here:
    # '--network none'
    #    entails there is no internet connection
    # 'gpus all'
    #    enables access to any GPUs present
    # '--volume <NAME>:/tmp'
    #   is added because on Grand Challenge this directory cannot be used to store permanent files
    # '-volume ../model:/opt/ml/model/":ro'
    #   is added to provide access to the (optional) tarball-upload locally
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        --gpus all \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}


run_docker_forward_pass "interface_0"



echo "=+= Save this image for uploading via ./do_save.sh"
