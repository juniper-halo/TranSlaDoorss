require("jest-canvas-mock");
<<<<<<< HEAD:frontend/tests/scripts.test.js
const { toggleCamera, requestTranslation, captureFrame } = require("../js/scripts");
=======

// Import scripts module
const scripts = require("./scripts");
const { toggleCamera, requestTranslation, captureFrame } = scripts;
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:js/scripts.test.js

// Mock the DOM elements before tests - they don't run on the actual webpage
beforeEach(() => {
  document.body.innerHTML = `
    <video id="webcam"></video>
    <button id="startCameraBtn">Start Camera</button>
    <input type="checkbox" id="showConfidence" />
    <textarea id="translationResult" readonly></textarea>
    <div id="feedbackSection" style="display: none;">
      <img id="capturedImage" src="" alt="Captured frame" />
      <button id="correctBtn" disabled>That's correct!</button>
      <button id="incorrectBtn" disabled>That's not right</button>
      <input type="text" id="correctLabel" disabled />
    </div>
  `;

  // Mock navigator.mediaDevices
  if (!navigator.mediaDevices) {
    navigator.mediaDevices = {};
  }

  // Mock video dimensions for captureFrame
  const video = document.getElementById("webcam");
  Object.defineProperty(video, "videoWidth", { value: 640, writable: true });
  Object.defineProperty(video, "videoHeight", { value: 480, writable: true });

  // Mock URL.createObjectURL
  global.URL.createObjectURL = jest.fn(() => "blob:mock-url");
});

test("toggleCamera starts and stops the webcam", async () => {
  // Mock webcam stream
  const mockStream = {
    // Generic javascript object
    getTracks: () => [
      {
        // Inline instance method returns an array of one generic object
        stop: jest.fn(), // This object, like a real media track,
        // has a stop function that is really just  a stub jest.fn()
      },
    ],
  };
  navigator.mediaDevices.getUserMedia = jest.fn().mockResolvedValue(mockStream);
  const video = document.getElementById("webcam");
  const btn = document.getElementById("startCameraBtn");
  // Start the camera
  await toggleCamera();
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
    video: true,
  });
  expect(video.srcObject).toBe(mockStream);
  expect(btn.textContent).toBe("Stop Camera");
  // Stop the camera
  await toggleCamera();
  expect(video.srcObject).toBeNull();
  expect(btn.textContent).toBe("Start Camera");
});

test("requestTranslation alerts if camera not started", async () => {
  window.alert = jest.fn();
  await requestTranslation();
  expect(window.alert).toHaveBeenCalledWith("Please start the camera first!");

  const mockStream = {
    getTracks: () => [
      {
        stop: jest.fn(),
      },
    ],
  };

  navigator.mediaDevices.getUserMedia = jest.fn().mockResolvedValue(mockStream);
  const video = document.getElementById("webcam");
  const btn = document.getElementById("startCameraBtn");
  await toggleCamera();
  await toggleCamera();

  await requestTranslation();
  expect(window.alert).toHaveBeenCalledWith("Please start the camera first!");
  expect(window.alert).toHaveBeenCalledTimes(2);
});

test("captureFrame captures a frame", async () => {
  const video = document.getElementById("webcam");
  video.videoWidth = 640;
  video.videoHeight = 480;

  const blob = await captureFrame();

  expect(blob).toBeInstanceOf(Blob);
  expect(blob.type).toBe("image/png");
});

test("requestTranslation send frame and update translation result", async () => {
  // Verify translation box is initially empty and read-only
  const translationBox = document.getElementById("translationResult");
  expect(translationBox.value).toBe("");
  expect(translationBox.hasAttribute("readonly")).toBe(true);

  const mockStream = {
    getTracks: () => [
      {
        stop: jest.fn(),
      },
    ],
  };
  navigator.mediaDevices.getUserMedia = jest.fn().mockResolvedValue(mockStream);
  const video = document.getElementById("webcam");
  const btn = document.getElementById("startCameraBtn");
  await toggleCamera();
  // Mock captureFrame to return a dummy blob
  const dummyBlob = new Blob(["dummy image data"], { type: "image/png" });
  window.captureFrame = jest.fn().mockResolvedValue(dummyBlob);

  // Mock fetch - return a fake successful response with new prediction format
  global.fetch = jest.fn().mockResolvedValue({
    json: () =>
      Promise.resolve({
        prediction: { letter: "A", confidence: 0.95 },
      }),
  });

  // Call the function
  await requestTranslation();

  // Verify fetch was called correctly
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch).toHaveBeenCalledWith("/img_in/translate/", {
    method: "POST",
    body: expect.any(FormData),
  });

  // Inspect the FormData that was sent
  const formData = fetch.mock.calls[0][1].body;
  expect(formData.get("language")).toBe("ASL");
  expect(formData.get("image")).toBeInstanceOf(Blob);
  expect(formData.get("image").name).toBe("frame.png");

  // Verify translation result is updated
  expect(translationBox.value).toBe("A");

  // Try again with a different translation
  fetch.mockResolvedValueOnce({
    json: () =>
      Promise.resolve({
        prediction: { letter: "B", confidence: 0.87 },
      }),
  });
  await requestTranslation();
  expect(translationBox.value).toBe("B");

  expect(fetch).toHaveBeenCalledTimes(2);
});
