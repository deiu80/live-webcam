const video = document.getElementById('video')
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri('/live-webcam/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/live-webcam/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/live-webcam/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/live-webcam/models')
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options()).withFaceExpressions()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    faceapi.draw.drawDetections(canvas,resizedDetections)
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
  }, 600)
})