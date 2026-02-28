// Class names for Garbage Detection (12 classes)
const labels = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
];

let session;
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
let isRunning = false;

// Load the ONNX model
async function loadModel() {
    try {
        statusDiv.innerText = "Loading model (Latest CDN)...";
        
        // Reset flags for latest version
        // ort.env.wasm.numThreads = 1; // Let it decide
        // ort.env.wasm.simd = true;   // Try default

        // Session options
        const options = {
            executionProviders: ['wasm'], // or 'webgl'
            graphOptimizationLevel: 'all'
        };

        // Retry fetch approach for better error handling
        const response = await fetch('./model.onnx');
        if (!response.ok) throw new Error("HTTP " + response.status + " " + response.statusText);
        const buffer = await response.arrayBuffer();

        // Create session
        session = await ort.InferenceSession.create(buffer, options);
        
        statusDiv.innerText = "Model loaded! Ready to detect.";
        console.log("Model loaded successfully");
    } catch (e) {
        console.error(e);
        const errorMsg = e.message || e.toString() || "Unknown error";
        statusDiv.innerText = "Failed to load model: " + errorMsg + ". Check console for details.";
        
        // Try to fetch model to see if it exists
        fetch('./model.onnx').then(response => {
             if (!response.ok) {
                 statusDiv.innerText += ` (HTTP ${response.status} fetching model.onnx)`;
             }
        });
    }
}

// Start the webcam stream
async function startCamera() {
    if (isRunning) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isRunning = true;
            statusDiv.innerText = "Running detection...";
            detectFrame();
        };
    } catch (e) {
        console.error(e);
        statusDiv.innerText = "Camera error: " + e.message;
    }
}

function stopCamera() {
    isRunning = false;
    const stream = video.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    statusDiv.innerText = "Camera stopped.";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Preprocessing: Resize and normalize image data
// YOLO typically expects [1, 3, 224, 224] depending on export size.
async function preprocess(videoElement) {
    const inputSize = 224; 
    
    // Draw video frame to an offscreen canvas to get pixel data
    const offscreen = document.createElement('canvas');
    offscreen.width = inputSize;
    offscreen.height = inputSize;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(videoElement, 0, 0, inputSize, inputSize);
    
    const imageData = offCtx.getImageData(0, 0, inputSize, inputSize);
    const { data, width, height } = imageData;
    
    // Convert to Float32 array in BCHW format [1, 3, 224, 224]
    // Also normalize 0-255 -> 0.0-1.0
    const float32Data = new Float32Array(1 * 3 * width * height);
    
    for (let i = 0; i < width * height; ++i) {
        // Red
        float32Data[i] = data[i * 4] / 255.0;
        // Green
        float32Data[width * height + i] = data[i * 4 + 1] / 255.0;
        // Blue
        float32Data[2 * width * height + i] = data[i * 4 + 2] / 255.0; 
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, width, height]);
    return { 
        // Try common YOLOv8 input names ('images' is standard, but check model if needed)
        images: tensor 
    };
}

// Main detection loop
async function detectFrame() {
    if (!isRunning || !session) return;
    
    // Fix: Define startTime here to avoid ReferenceError
    const startTime = performance.now();

    try {
        // console.log("Starting inference..."); // Debug print
        const feeds = await preprocess(video);
        const results = await session.run(feeds);
        
        const outputName = session.outputNames[0];
        const outputTensor = results[outputName];
        const output = outputTensor.data; 
        const dims = outputTensor.dims || [1, 4 + labels.length, output.length/(4 + labels.length)]; // Fallback
        
        const boxes = processOutput(output, dims, video.videoWidth, video.videoHeight);
        drawBoxes(boxes);
        
        // Calculate FPS
        const endTime = performance.now();
        const fps = 1000 / (endTime - startTime);
        
        // Optional: Update debug UI
        /*
        const debugDiv = document.getElementById('debug');
        if (debugDiv && Math.random() < 0.1) {
             debugDiv.innerText = `FPS: ${fps.toFixed(1)} | Dims: ${dims}`;
        }
        */

    } catch (e) {
        console.error(e);
        stopCamera();
        statusDiv.innerText = "Inference error: " + e.message;
    }
    
    if (isRunning) requestAnimationFrame(detectFrame);
}

function processOutput(output, dims, imgWidth, imgHeight) {
    let numClass = 12; // 12 classes
    let numElements = 4 + numClass; // 16 elements per anchor
    let numAnchors = 0;
    
    // Determine shape (Planar [1, 16, N] vs Transposed [1, N, 16])
    let isTransposed = false;
    
    if (dims.length === 3) {
        if (dims[1] === numElements) {
            numAnchors = dims[2]; // Standard [1, 16, 1029]
        } else if (dims[2] === numElements) {
            numAnchors = dims[1]; // Transposed [1, 1029, 16]
            isTransposed = true;
        } else {
            // Fallback assumption based on length if dims don't match 16 exactly
            // (e.g. if batch size > 1, though unlikely here)
            numAnchors = output.length / numElements;
        }
    } else {
         numAnchors = output.length / numElements;
    }

    let boxes = [];
    let maxScoreFound = 0;

    for (let i = 0; i < numAnchors; i++) {
        let cx, cy, w, h, maxConf = 0, maxClass = -1;

        if (!isTransposed) {
            // Planar: [1, 16, N]
            // Channel 0 is all cx's. Channel 1 is all cy's.
            cx = output[0 * numAnchors + i];
            cy = output[1 * numAnchors + i];
            w  = output[2 * numAnchors + i];
            h  = output[3 * numAnchors + i];
            
            for (let c = 0; c < numClass; c++) {
                const conf = output[(4 + c) * numAnchors + i];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClass = c;
                }
            }
        } else {
            // Transposed: [1, N, 16]
            // Stride is 16
            const offset = i * numElements;
            cx = output[offset + 0];
            cy = output[offset + 1];
            w  = output[offset + 2];
            h  = output[offset + 3];
            
            for (let c = 0; c < numClass; c++) {
                const conf = output[offset + 4 + c];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClass = c;
                }
            }
        }
        
        if (maxConf > maxScoreFound) maxScoreFound = maxConf;

        if (maxConf > 0.20) { 
            // The debug output (w=2.9, h=2.0) suggests the model output is NORMALIZED (0-1)
            // or relative to a very small grid (not 224).
            // Let's assume it is NORMALIZED (0.0-1.0) first.
            // If cx, cy, w, h are 0.0-1.0:
            
            const x = (cx - w / 2) * imgWidth;
            const y = (cy - h / 2) * imgHeight;
            const width = w * imgWidth;
            const height = h * imgHeight;
            
            boxes.push({ x, y, width, height, class: maxClass, score: maxConf });
        }
    }
    
    // Debug logging (throttled 20%)
    if (Math.random() < 0.2 && boxes.length > 0) {
        // Log the first candidate box
        console.log(`Debug - Box0: x=${boxes[0].x.toFixed(1)}, y=${boxes[0].y.toFixed(1)}, w=${boxes[0].width.toFixed(1)}, h=${boxes[0].height.toFixed(1)} | Score: ${boxes[0].score.toFixed(2)}`);
    }

    return nms(boxes);
}

function nms(boxes) {
    if (boxes.length === 0) return [];
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const best = boxes.shift();
        result.push(best);
        boxes = boxes.filter(box => {
            const iou = calculateIoU(best, box);
            return iou < 0.45;
        });
    }
    return result;
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = (a.width * a.height) + (b.width * b.height) - intersection;
    return intersection / union;
}

function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 4;
    ctx.font = "20px Arial";
    
    boxes.forEach(box => {
        const color = getColor(box.class);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        
        const label = `${labels[box.class]} (${Math.round(box.score * 100)}%)`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(box.x, box.y - 25, textWidth + 10, 25);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, box.x + 5, box.y - 5);
    });
}

function getColor(index) {
    const colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33F6", "#F6FF33", "#33FFF6", "#FF8C33", "#8C33FF", "#33FF8C", "#FF338C", "#438CFF"];
    return colors[index % colors.length];
}

// Start loading the model when the page loads
if (document.readyState === 'complete') {
    loadModel(); 
} else {
    window.addEventListener('load', loadModel);
}
