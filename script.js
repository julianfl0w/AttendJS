const video = document.getElementById("video");

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(startWebcam);

function startWebcam() {
  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    video.srcObject = stream;
  }).catch((err) => {
    console.error("Error accessing webcam", err);
  });
}

async function getLabeledFaceDescriptions() {
  // Load the manifest
  const response = await fetch('manifest.json');
  const manifest = await response.json();

  // Parse the manifest and process each name and associated images
  const labeledFaceDescriptors = await Promise.all(manifest.map(async (dir) => {
    if (dir.type !== "directory" || !dir.contents) return;

    const label = dir.name; // Assuming the directory name is the label
    const descriptions = [];

    for (let item of dir.contents) {
      if (item.type !== "file") continue;

      const img = await faceapi.fetchImage(`./labels/${label}/${item.name}`);
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
      if (detections) descriptions.push(detections.descriptor);
    }

    return new faceapi.LabeledFaceDescriptors(label, descriptions);
  }));

  return labeledFaceDescriptors.filter(descriptor => descriptor); // Filter out any undefined entries
}

video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
      drawBox.draw(canvas);
    });
  }, 100);
});
