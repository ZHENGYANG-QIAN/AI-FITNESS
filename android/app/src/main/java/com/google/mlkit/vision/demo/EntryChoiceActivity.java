package com.google.mlkit.vision.demo;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.mlkit.vision.demo.java.ChooserActivity;

import java.util.ArrayList;
import java.util.List;

public class EntryChoiceActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final String TAG = "EntryChoiceActivity";
    private static final int PERMISSION_REQUESTS = 1;
    private static final String[] REQUIRED_RUNTIME_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = new Intent(EntryChoiceActivity.this, ChooserActivity.class);
        startActivity(intent);

        if (!allRuntimePermissionsGranted()) {
            getRuntimePermissions();
        }
    }

    private boolean allRuntimePermissionsGranted() {
        for (String permission : REQUIRED_RUNTIME_PERMISSIONS) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> permissionsToRequest = new ArrayList<>();
        for (String permission : REQUIRED_RUNTIME_PERMISSIONS) {
            if (!isPermissionGranted(this, permission)) {
                permissionsToRequest.add(permission);
            }
        }

        if (!permissionsToRequest.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest.toArray(new String[0]),
                    PERMISSION_REQUESTS
            );
        }
    }

    private boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }
}