/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.common.annotation.KeepName;
import com.google.mlkit.vision.demo.CameraSource;
import com.google.mlkit.vision.demo.CameraSourcePreview;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.demo.java.facedetector.FaceDetectorProcessor;
import com.google.mlkit.vision.demo.java.posedetector.PoseDetectorProcessor;
import com.google.mlkit.vision.demo.java.segmenter.SegmenterProcessor;
import com.google.mlkit.vision.demo.preference.SettingsActivity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Live preview demo for ML Kit APIs.
 */
@KeepName
public final class LivePreviewActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final String TAG = "LivePreviewActivity";

    private static final String POSE_DETECTION = "人体关键点检测";
    private static final String POSE_CLASSIFICATION = "健身动作分类/计数";
    private static final String SELFIE_SEGMENTATION = "人像抠图";
    private static final String FACE_DETECTION = "人脸检测";

    private static final int PERMISSION_REQUESTS = 1;

    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;
    private String selectedMode = POSE_DETECTION;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_vision_live_preview);

        preview = findViewById(R.id.preview_view);
        if (preview == null) {
            Log.d(TAG, "Preview is null");
        }
        graphicOverlay = findViewById(R.id.graphic_overlay);
        if (graphicOverlay == null) {
            Log.d(TAG, "graphicOverlay is null");
        }

        populateFeatureSelector();
        populateFaceSwitch();

        ImageView settingsButton = findViewById(R.id.settings_button);
        settingsButton.setOnClickListener(
                v -> {
                    Intent intent = new Intent(getApplicationContext(), SettingsActivity.class);
                    intent.putExtra(
                            SettingsActivity.EXTRA_LAUNCH_SOURCE, SettingsActivity.LaunchSource.LIVE_PREVIEW);
                    startActivity(intent);
                });

        createCameraSource(selectedMode);
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        createCameraSource(selectedMode);
        startCameraSource();
    }

    /**
     * Stops the camera.
     */
    @Override
    protected void onPause() {
        super.onPause();
        preview.stop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraSource != null) {
            cameraSource.release();
        }
    }

    private void populateFeatureSelector() {
        Spinner spinner = findViewById(R.id.spinner);
        List<String> options = new ArrayList<>();
        options.add(POSE_DETECTION);
        options.add(POSE_CLASSIFICATION);
        options.add(SELFIE_SEGMENTATION);
        options.add(FACE_DETECTION);

        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<>(this, R.layout.spinner_style, options);
        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // attaching data adapter to spinner
        spinner.setAdapter(dataAdapter);
        spinner.setOnItemSelectedListener(
                new OnItemSelectedListener() {

                    @Override
                    public synchronized void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                        // An item was selected. You can retrieve the selected item using
                        // parent.getItemAtPosition(pos)
                        selectedMode = parent.getItemAtPosition(pos).toString();
                        Log.d(TAG, "Selected model: " + selectedMode);
                        preview.stop();
                        createCameraSource(selectedMode);
                        startCameraSource();
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {
                        // Do nothing.
                    }
                });
    }

    private void populateFaceSwitch() {
        ToggleButton facingSwitch = findViewById(R.id.facing_switch);
        facingSwitch.setOnCheckedChangeListener(
                (buttonView, isChecked) -> {
                    Log.d(TAG, "Set facing");
                    if (cameraSource != null) {
                        if (isChecked) {
                            cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
                        } else {
                            cameraSource.setFacing(CameraSource.CAMERA_FACING_BACK);
                        }
                    }
                    preview.stop();
                    startCameraSource();
                }
        );
    }

    private void createCameraSource(String model) {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        try {
            switch (model) {
                case POSE_DETECTION:
                    cameraSource.setMachineLearningFrameProcessor(
                            new PoseDetectorProcessor(
                                    this,
                                    /* runClassification =*/ false,
                                    /* isStreamMode = */ true));
                    break;
                case POSE_CLASSIFICATION:
                    cameraSource.setMachineLearningFrameProcessor(
                            new PoseDetectorProcessor(
                                    this,
                                    /* runClassification =*/ true,
                                    /* isStreamMode = */ true));
                    break;
                case SELFIE_SEGMENTATION:
                    cameraSource.setMachineLearningFrameProcessor(new SegmenterProcessor(this));
                    break;
                case FACE_DETECTION:
                    Log.i(TAG, "Using Face Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new FaceDetectorProcessor(this));
                    break;
                default:
                    Log.e(TAG, "Unknown model: " + model);
            }
        } catch (RuntimeException e) {
            Log.e(TAG, "Can not create image processor: " + model, e);
            Toast.makeText(
                            getApplicationContext(),
                            "创建图像处理器失败: " + e.getMessage(),
                            Toast.LENGTH_LONG)
                    .show();
        }
    }

    /**
     * Starts or restarts the camera source, if it exists. If the camera source doesn't exist yet
     * (e.g., because onResume was called before the camera source was created), this will be called
     * again when the camera source is created.
     */
    private void startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d(TAG, "resume: Preview is null");
                }
                if (graphicOverlay == null) {
                    Log.d(TAG, "resume: graphOverlay is null");
                }
                preview.start(cameraSource, graphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Unable to start camera source.", e);
                cameraSource.release();
                cameraSource = null;
            }
        }
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    this.getPackageManager()
                            .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> allNeededPermissions = new ArrayList<>();
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                allNeededPermissions.add(permission);
            }
        }

        if (!allNeededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this, allNeededPermissions.toArray(new String[0]), PERMISSION_REQUESTS);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        Log.i(TAG, "Permission granted!");
        if (allPermissionsGranted()) {
            createCameraSource(selectedMode);
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private static boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }
}
