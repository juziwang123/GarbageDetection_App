# Android Garbage Detection App

This folder structure is prepared for an Android Studio project.

## Setup Instructions

1.  **Open in Android Studio**: Open the `android_app` folder (or create a new project and copy these files).
2.  **Add Dependencies**: In your `build.gradle` (Module: app), add:
    ```gradle
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.4.4'
    ```
3.  **Model Placement**: Place your trained `yolov8n_garbage_float16.tflite` (or similar name) into `app/src/main/assets/`.
4.  **Permissions**: Ensure `AndroidManifest.xml` has camera permissions:
    ```xml
    <uses-permission android:name="android.permission.CAMERA" />
    ```

## Key Components

-   **MainActivity.kt**: Handles camera preview and inference loop.
-   **Detector.kt**: Wrapper class for TFLite interpreter.
-   **OverlayView.kt**:  (Optional) Custom view to draw bounding boxes on top of the camera preview.
