/*!
 * Start Bootstrap - Grayscale v7.0.6 (https://startbootstrap.com/theme/grayscale)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
 */
//
// Scripts
//

// Webcam stream reference
let webcamStream = null;

// Store last captured frame and prediction for feedback
let lastCapturedBlob = null;
let lastPrediction = null;

// Toggle the webcam on/off
async function toggleCamera() {
  const video = document.getElementById("webcam");
  const btn = document.getElementById("startCameraBtn");

  if (webcamStream) {
    // Stop the camera
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
    video.srcObject = null;
    btn.textContent = "Start Camera";
    console.log("Camera stopped!");
  } else {
    // Start the camera
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = webcamStream;
      btn.textContent = "Stop Camera";
      console.log("Camera started!");
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please allow camera permissions.");
    }
  }
}

// Placeholder function for translation requests
async function requestTranslation() {
  console.log("Translation requested!");
  // TODO: Capture frame from video and send to backend for translation
  if (!webcamStream) {
    alert("Please start the camera first!");
    return;
  }

  const frameBlob = await captureFrame();

  // Store the captured frame for feedback
  lastCapturedBlob = frameBlob;

  const formData = new FormData();
  formData.append("image", frameBlob, "frame.png");
  formData.append("language", "ASL");
  const response = await fetch("/img_in/translate/", {
    method: "POST",
    body: formData,
  });
  const result = await response.json();

  // Store prediction for feedback
  lastPrediction = result.prediction;

  const translationBox = document.getElementById("translationResult");
  const showConfidence = document.getElementById("showConfidence").checked;

  if (showConfidence) {
    const confidencePercent = (result.prediction.confidence * 100).toFixed(1);
    translationBox.value = `${result.prediction.letter} (${confidencePercent}% confidence)`;
  } else {
    translationBox.value = result.prediction.letter;
  }

  // Show feedback section with captured image
  showFeedbackSection();
}

// Capture a frame from the webcam video
async function captureFrame() {
  const video = document.getElementById("webcam");
  const canvas = document.createElement("canvas");

  // Set canvas size to match video dimensions
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw the current video frame onto the canvas
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  // A blob is a file-like object that can be sent via FormData
  // Unfortunately this is done by asynchronous callback
  const blob = await new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/png"); // Image format is set to PNG
  });

  return blob;
}

// Show feedback section with captured image
function showFeedbackSection() {
  const feedbackSection = document.getElementById("feedbackSection");
  const capturedImage = document.getElementById("capturedImage");
  const correctBtn = document.getElementById("correctBtn");
  const incorrectBtn = document.getElementById("incorrectBtn");
  const correctLabel = document.getElementById("correctLabel");

  if (!lastCapturedBlob) {
    console.error("No captured image available for feedback");
    return;
  }

  try {
    // Convert blob to data URL for display
    const imageUrl = URL.createObjectURL(lastCapturedBlob);
    capturedImage.src = imageUrl;

    // Show section and enable buttons
    feedbackSection.style.display = "block";
    correctBtn.disabled = false;
    incorrectBtn.disabled = false;
    correctLabel.disabled = false;
    correctLabel.value = "";
  } catch (error) {
    console.error("Error showing feedback section:", error);
  }
}

// Submit feedback indicating prediction was correct
async function submitCorrectFeedback() {
  await submitFeedback(lastPrediction.letter);
}

// Submit feedback with user-provided correct label
async function submitIncorrectFeedback() {
  const inputValue = document.getElementById("correctLabel").value;
  const correctLabel = inputValue.toUpperCase();

  if (
    !correctLabel ||
    correctLabel.length !== 1 ||
    !/[A-Z]/.test(correctLabel)
  ) {
    alert("Please enter a single uppercase letter (A-Z)");
    return;
  }

  await submitFeedback(correctLabel);
}

// Send feedback to backend
async function submitFeedback(correctLabel) {
  const formData = new FormData();
  formData.append("image", lastCapturedBlob, "feedback.png");
  formData.append("predicted_label", lastPrediction.letter);
  formData.append("correct_label", correctLabel);

  try {
    const response = await fetch("/img_in/feedback/", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      alert("Thank you for your feedback!");
      resetFeedbackSection();
    } else {
      const errorText = await response.text();
      console.error("Feedback submission failed:", response.status, errorText);
      alert(
        `Failed to submit feedback (${response.status}). Check console for details.`,
      );
    }
  } catch (err) {
    console.error("Error submitting feedback:", err);
    alert("Failed to submit feedback. Please try again.");
  }
}

// Reset feedback section after submission
function resetFeedbackSection() {
  const feedbackSection = document.getElementById("feedbackSection");
  const correctBtn = document.getElementById("correctBtn");
  const incorrectBtn = document.getElementById("incorrectBtn");
  const correctLabel = document.getElementById("correctLabel");

  feedbackSection.style.display = "none";
  correctBtn.disabled = true;
  incorrectBtn.disabled = true;
  correctLabel.disabled = true;
  correctLabel.value = "";

  lastCapturedBlob = null;
  lastPrediction = null;
}

window.addEventListener("DOMContentLoaded", (event) => {
  // Navbar shrink function
  var navbarShrink = function () {
    const navbarCollapsible = document.body.querySelector("#mainNav");
    if (!navbarCollapsible) {
      return;
    }
    if (window.scrollY === 0) {
      navbarCollapsible.classList.remove("navbar-shrink");
    } else {
      navbarCollapsible.classList.add("navbar-shrink");
    }
  };

  // Shrink the navbar
  navbarShrink();

  // Shrink the navbar when page is scrolled
  document.addEventListener("scroll", navbarShrink);

  // Activate Bootstrap scrollspy on the main nav element
  const mainNav = document.body.querySelector("#mainNav");
  if (mainNav) {
    new bootstrap.ScrollSpy(document.body, {
      target: "#mainNav",
      rootMargin: "0px 0px -40%",
    });
  }

  // Collapse responsive navbar when toggler is visible
  const navbarToggler = document.body.querySelector(".navbar-toggler");
  const responsiveNavItems = [].slice.call(
    document.querySelectorAll("#navbarResponsive .nav-link"),
  );
  responsiveNavItems.map(function (responsiveNavItem) {
    responsiveNavItem.addEventListener("click", () => {
      if (window.getComputedStyle(navbarToggler).display !== "none") {
        navbarToggler.click();
      }
    });
  });
});

// Export functions for testing (ignored by browsers)
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    toggleCamera,
    requestTranslation,
    captureFrame,
    showFeedbackSection,
    submitCorrectFeedback,
    submitIncorrectFeedback,
  };
}
