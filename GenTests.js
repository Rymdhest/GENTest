const IMAGE_H = 28;
const IMAGE_W = 28;
const MODEL_INPUT_SIZE = 299;
const GENERATOR_INPUT_SIZE = 100;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const SAMPLE_SIZE = 1000 // how many images to use in score calculations
const MNIST_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

run();
// Main function to execute generation, FID, and IS calculation
async function run() {
    const generator = createGenerator();
    const model = await tf.loadLayersModel('http://localhost:5000/static/InceptionV3/model.json');

    // Generate images
    const noise = tf.randomNormal([SAMPLE_SIZE, GENERATOR_INPUT_SIZE]);
    const generatedImages = generator.predict(noise, { training: false });
    realImages = await loadMnistData()

    // Calculate FID and IS
    const fid = await calculateFID(realImages.slice(0, SAMPLE_SIZE), generatedImages, model);
    const isScore = await calculateIS(generatedImages, model);

    plotImages(realImages, 'Original MNIST Digits');
    plotImages(generatedImages, 'Generated MNIST Digits');

    document.getElementById('loading').textContent = '';

    document.getElementById('FID').textContent = `FID Score: ${fid}`;
    document.getElementById('IS').textContent = `Inception Score: ${isScore}`;
}

// Define a simple generator
function createGenerator() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 7 * 7 * 256, useBias: false, inputShape: [GENERATOR_INPUT_SIZE], kernelInitializer: 'randomNormal' }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU());
    model.add(tf.layers.reshape({ targetShape: [7, 7, 256] }));
    model.add(tf.layers.conv2dTranspose({ filters: 128, kernelSize: 5, strides: [1, 1], padding: 'same', useBias: false, kernelInitializer: 'randomNormal' }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU());
    model.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: 5, strides: [2, 2], padding: 'same', useBias: false, kernelInitializer: 'randomNormal' }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU());
    model.add(tf.layers.conv2dTranspose({ filters: 1, kernelSize: 5, strides: [2, 2], padding: 'same', useBias: false, activation: 'tanh', kernelInitializer: 'randomNormal' }));
    return model;
}

// Resize and convert from grayscale to RGB
async function preprocessImages(inputImages) {
    resizedImages = tf.image.resizeBilinear(inputImages, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    rgbImages = tf.concat([resizedImages, resizedImages, resizedImages], 3);
    resizedImages.dispose()
    return rgbImages;
}


// Function to calculate FID
async function calculateFID(realImages, generatedImages, inceptionModel) {
    const preprocessedRealImages = await preprocessImages(realImages);

    const realActivations = inceptionModel.predict(preprocessedRealImages);
    preprocessedRealImages.dispose()

    const realActivationsArray = await realActivations.array();
    realActivations.dispose()

    const preprocessedGeneratedImages = await preprocessImages(generatedImages);

    const generatedActivations = inceptionModel.predict(preprocessedGeneratedImages);
    preprocessedGeneratedImages.dispose()

    const generatedActivationsArray = await generatedActivations.array();
    generatedActivations.dispose()

    // Calculate mean and covariance
    const mu1 = math.mean(realActivationsArray, 0);
    const sigma1 = await calculateRemote(realActivationsArray, 'covariance');
    const mu2 = math.mean(generatedActivationsArray, 0);
    const sigma2 = await calculateRemote(generatedActivationsArray, 'covariance');

    // Calculate sum squared difference between means
    const ssdiff = math.sum(math.map(math.subtract(mu1, mu2), (value) => math.square(value)));

    // Calculate sqrt of product between covariances
    const covmean = await calculateRemote(math.multiply(sigma1, sigma2), 'sqrtm');

    // Calculate FID
    const fid = ssdiff + math.trace(tf.add(sigma1, sigma2).sub(tf.mul(covmean, 2)).arraySync());
    return fid;
}

async function calculateIS(generatedImages, inceptionModel) {
    const preprocessedImages = await preprocessImages(generatedImages);
    const pYX = inceptionModel.predict(preprocessedImages);
    preprocessedImages.dispose()

    const pY = pYX.mean(0).expandDims(0);
    const klD = pYX.mul(pYX.add(1e-16).log().sub(pY.add(1e-16).log()));
    pYX.dispose()
    pY.dispose()
    const isScore = tf.exp(klD.sum(1).mean()).dataSync()[0];
    return isScore;
}

// Load MNIST data
async function loadMnistData() {
    const img = new Image();
    nTrain = 60000
    img.crossOrigin = '';

    // Load the images from the sprite
    const imgPromise = new Promise((resolve) => {
        img.onload = () => {
            const datasetBytesBuffer = new ArrayBuffer(nTrain * IMAGE_SIZE * 4);
            const chunkSize = 5000;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = img.width;
            canvas.height = chunkSize;

            for (let i = 0; i < nTrain / chunkSize; i++) {
                const datasetBytesView = new Float32Array(
                    datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                    IMAGE_SIZE * chunkSize);
                ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                for (let j = 0; j < imageData.data.length / 4; j++) {
                    datasetBytesView[j] = (imageData.data[j * 4] / 255) * 2 - 1; // map from 0,255 to -1,1
                }
            }
            resolve(new Float32Array(datasetBytesBuffer));
        };
        img.src = MNIST_URL;
    });
    const datasetImages = await imgPromise;
    const trainImages = datasetImages.slice(0, IMAGE_SIZE * nTrain);// Slice the images into the training set

    // Create tensor from training images
    const x_train = tf.tensor4d(trainImages, [trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);

    return x_train;
}

// Function to display images
function plotImages(images, title, numImages = 10) {
    const container = document.getElementById('imageContainer');
    const originalTitle = document.createElement('div');
    originalTitle.className = 'image-title';
    originalTitle.innerText = title;
    container.appendChild(originalTitle);

    for (let i = 0; i < numImages; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = IMAGE_W;
        canvas.height = IMAGE_H;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(IMAGE_W, IMAGE_H);
        const data = images.slice([i, 0, 0, 0], [1, IMAGE_W, IMAGE_H, 1]).reshape([IMAGE_W, IMAGE_H]).dataSync();
        for (let j = 0; j < data.length; j++) {
            const value = Math.floor(((data[j] + 1) / 2) * 255); //rescale it from [-1, 1] to [0, 255]
            imageData.data[4 * j] = value;       // Red channel
            imageData.data[4 * j + 1] = value;   // Green channel
            imageData.data[4 * j + 2] = value;   // Blue channel
            imageData.data[4 * j + 3] = 255;     // Alpha channel
        }
        ctx.putImageData(imageData, 0, 0);
        container.appendChild(canvas);
    }
}

// helper function to use certain functions from a local python server
async function calculateRemote(input, route) {
    const data = {
        data:
            input
    };

    try {
        const response = await fetch('http://localhost:5000/' + route, {  // Use absolute URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            return result.result;  // Return the result matrix
        } else {
            console.log(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}