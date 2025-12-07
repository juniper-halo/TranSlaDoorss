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

// Toggle the webcam on/off
async function toggleCamera() {
  const video = document.getElementById("webcam");
  const btn = document.getElementById("startCameraBtn");

  if (webcamStream) {
    // Stop the camera
    webcamStream.getTracks().forEach(track => track.stop());
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
  const formData = new FormData();
  formData.append("image", frameBlob, "frame.png");
  formData.append("language", "ASL");
  const response = await fetch("/img_in/translate/", {
    method: "POST",
    body: formData,
  });
  const result = await response.json();
  console.log("Translation result:", result);
  const translationBox = document.getElementById("translationResult");
  translationBox.value = result.translation;
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
  module.exports = { toggleCamera, requestTranslation, captureFrame };
}
