let warn = document.getElementById("model_log");
// let enableWebcamButton = document.getElementById("webcamButton");
let emotionText = document.getElementById("emotion");

const emotions = ["Angry", "Happy", "Sad", "Surprise"];
var tfliteModel = undefined;

async function start() {
    await tflite.loadTFLiteModel(
        "static/model.tflite"
    ).then((loadedModel) => { 
        tfliteModel = loadedModel;
        warn.innerHTML = "Model has successfully loaded! Your camera should be displayed soon."
        // enableWebcamButton.classList.remove("invisible");
    });
}
start();

function openCvReady() {
    cv['onRuntimeInitialized'] = () => {
        let video = document.getElementById("cam_input"); // video is the id of video tag
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function (stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function (err) {
                console.log("An error occurred! " + err);
            });

        let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let gray = new cv.Mat();
        let gray_roi = new cv.Mat();
        let cap = new cv.VideoCapture(cam_input);
        let faces = new cv.RectVector();
        let utils = new Utils('errorMessage');
        let classifier = new cv.CascadeClassifier();
        let faceCascadeFile = 'haarcascade_frontalface_default.xml';
        utils.createFileFromUrl(faceCascadeFile,
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", () => {
                classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
            });


        const FPS = 30;
        // emotionText.classList.remove("invisible");

        function processVideo() {
            let begin = Date.now();
            cap.read(src);
            src.copyTo(dst);
            cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
            
            // detect face(s)
            try {
                classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
            } catch (err) {
                console.log(err);
            }
            
            for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                
                // rect for gray roi
                let rect = new cv.Rect(face.x, face.y, face.width, face.height);
                
                // mat for resize gray roi
                gray_roi = gray.roi(rect);
                gray_roi_resize = new cv.Mat();
                cv.resize(gray_roi, gray_roi_resize, new cv.Size(244, 244))
                cv.imshow("canvas_roi", gray_roi_resize);
                
                // predict using model
                const outputTensor = tf.tidy(() => {
                    // Get pixels data from an image.
                    let img = tf.browser.fromPixels(document.getElementById("canvas_roi"));

                    // Resize, normalize, expand dimensions of image pixels by 0 axis.:
                    img = tf.image.resizeBilinear(img, [48, 48]);
                    img = tf.div(tf.expandDims(img, 0), 255);
                    
                    // predict
                    let outputTensor = tfliteModel.predict(img);
                    return outputTensor;
                });

                // convert to array and take prediction index with highest value
                let output = outputTensor.arraySync();
                let index = output[0].indexOf(Math.max(...output[0]));
                
                // render rectangles and text
                cv.rectangle(dst, point1, point2, [255, 0, 0, 255], 2);
                cv.putText(dst, emotions[index], new cv.Point(face.x, face.y),
                cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(0, 0, 255), 2);
                cv.imshow("canvas_output", dst);
                // emotionText.innerHTML(emotions[index]);
            }
            
            
            // schedule next one.
            let delay = 1000 / FPS - (Date.now() - begin);
            setTimeout(processVideo, delay);
        }

        setTimeout(processVideo, 0);
    };
}
