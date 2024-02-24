const video = document.getElementById("video");

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
]).then(startWebcam);

function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}
async function getLabeledFaceDescriptions() {
  // Load and parse the manifest file
  const response = await fetch('manifest.json');
  const manifestJson = await response.json();
  const namesDirectory = manifestJson.find(item => item.type === "directory" && item.name === "names");

  // Filter directories from the manifest
  var names = namesDirectory.contents.filter(item => item.type === "directory");

  // Iterate over each directory and its contents
  return Promise.all(names.map(async (dir) => {
    const label = dir.name; // The directory name is used as the label
    const descriptions = [];

    // Iterate over each file in the directory
    for (let item of dir.contents) {
      if (item.type !== "file") continue; // Skip if not a file

      const img = await faceapi.fetchImage(`./names/${label}/${item.name}`);
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      // Ensure detections were successful before adding
      if (detections) {
        descriptions.push(detections.descriptor);
      }
    }

    return new faceapi.LabeledFaceDescriptors(label, descriptions);
  }));
}


video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    const results = resizedDetections.map((d) => {
      return faceMatcher.findBestMatch(d.descriptor);
    });
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result,
      });
      drawBox.draw(canvas);
    });
  }, 100);
});
