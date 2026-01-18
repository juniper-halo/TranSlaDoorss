require("jest-canvas-mock");
const { toggleCamera, requestTranslation, captureFrame } = require("../js/scripts");

// Mock the DOM elements before tests - they don't run on the actual webpage
beforeEach(() => {
  document.body.innerHTML = `
    <video id="webcam"></video>
    <button id="startCameraBtn">Start Camera</button>
    <textarea id="translationResult" readonly></textarea>
  `;

  // Mock navigator.mediaDevices
  if (!navigator.mediaDevices) {
    navigator.mediaDevices = {};
  }
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

  // Mock fetch - return a fake successful response
  global.fetch = jest.fn().mockResolvedValue({
    json: () => Promise.resolve({ translation: "a" }),
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
  expect(translationBox.value).toBe("a");

  // Try again with a different translation
  fetch.mockResolvedValueOnce({
    json: () => Promise.resolve({ translation: "b" }),
  });
  await requestTranslation();
  expect(translationBox.value).toBe("b");

  expect(fetch).toHaveBeenCalledTimes(2);
});
