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
  const croppedCanvas = document.getElementById('face-canvas')
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }

  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    const faces = await faceapi.detectSingleFace(video, new faceapi.SsdMobilenetv1Options()).withFaceExpressions()
    const results = await faceapi.detectAllFaces(
      video,
      new faceapi.SsdMobilenetv1Options()
    ).withFaceExpressions();
   
    const resizedDetections = faceapi.resizeResults(results, displaySize)

    const box = {
    bottom: 0,
    left: 9000,
    right: 0,
    top: 9000,

    get height() {
      return this.bottom - this.top;
    },

    get width() {
      return this.right - this.left;
    },
  };
  for (const face of results) {
    box.bottom = Math.max(box.bottom, face.detection.box.bottom);
    box.left = Math.min(box.left, face.detection.box.left);
    box.right = Math.max(box.right, face.detection.box.right);
    box.top = Math.min(box.top, face.detection.box.top);
  }
  
  const resizedDetectionsCropped = faceapi.resizeResults(results, {width:box.width,height:box.height})

  
    
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

    croppedCanvas.getContext('2d').clearRect(0, 0, croppedCanvas.width, croppedCanvas.height)
    
   
    var myCTX = croppedCanvas.getContext('2d');
 
  
    myCTX.drawImage(video,
    box.left,
    box.top,
    box.width,
    box.height,
    0,
    0,
    box.width,
    box.height);

    faceapi.draw.drawFaceExpressions(croppedCanvas,resizedDetectionsCropped)
  
    // faceapi.draw.drawDetections(canvas,resizedDetections)
    // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)


  }, 100)
})

const labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
// Define the function to make predictions on the video frames
async function predict() {
 
  var myCanvasElement = document.getElementById('face-canvas')
  var myCTX = myCanvasElement.getContext('2d');
  myCTX.drawImage(video, 0, 0, myCanvasElement.width, myCanvasElement.height);

  // Convert the canvas image to a tensor
  const tensor = tf.browser.fromPixels(myCanvasElement)
  // the scalars needed for conversion of each channel
  // per the formula: gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
  rFactor = tf.scalar(0.2989);
  gFactor = tf.scalar(0.5870);
  bFactor = tf.scalar(0.1140);

  // separate out each channel. x.shape[0] and x.shape[1] will give you
  // the correct dimensions regardless of image size
  r = tensor.slice([0, 0, 0], [tensor.shape[0], tensor.shape[1], 1]);
  g = tensor.slice([0, 0, 1], [tensor.shape[0], tensor.shape[1], 1]);
  b = tensor.slice([0, 0, 2], [tensor.shape[0], tensor.shape[1], 1]);

  // add all the tensors together, as they should all be the same dimensions.
  gray = r.mul(rFactor).add(g.mul(gFactor)).add(b.mul(bFactor));

  const grayTensorImg = gray;

  // Resize the tensor to match the input size of the model
  const resizedTensor = tf.image.resizeBilinear(grayTensorImg, [48, 48]);

  // Normalize the tensor
  const normalizedTensor = tf.div(resizedTensor, 255);

  // Add a dimension to the tensor to match the batch size of the model
  const batchedTensor = normalizedTensor.expandDims(0);

  const model = await tf.loadLayersModel('/model.json')
  // Make a prediction on the tensor
  const prediction = await model.predict(batchedTensor);

  // Apply the softmax function to obtain a normalized probability distribution


  // Get the index of the class with the highest probability
  const predClass = prediction.dataSync();
  const predictedValueArr = prediction.arraySync()[0];

  var maxValue = Math.max(...predictedValueArr);
  var maxIndex = predictedValueArr.indexOf(maxValue);
  // Do something with the prediction (e.g. update the UI)

  myCTX.font = '20px Arial';
  myCTX.fillStyle = 'red';
  myCTX.fillText(`Predicted class: ${labels[maxIndex]} with ${maxValue}`, 10, 30);
}