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

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.os.StrictMode;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

import com.google.mlkit.vision.demo.BuildConfig;
import com.google.mlkit.vision.demo.R;

import java.util.ArrayList;
import java.util.List;

/**
 * Demo app chooser which allows you pick from all available testing Activities.
 */
public final class ChooserActivity extends AppCompatActivity
        implements AdapterView.OnItemClickListener {
    // RUNTIME_PERMISSIONS
    private static final String TAG = "ChooserActivity";
    private static final int PERMISSION_REQUESTS = 1;
    private static final String[] REQUIRED_RUNTIME_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
    };

    // Activities
    @SuppressWarnings("NewApi") // CameraX is only available on API 21+
    private static final Class<?>[] CLASSES =
            VERSION.SDK_INT < VERSION_CODES.LOLLIPOP
                    ? new Class<?>[]{
                    LivePreviewActivity.class, StillImageActivity.class,
            }
                    : new Class<?>[]{
                    LivePreviewActivity.class,
                    StillImageActivity.class,
                    CameraXLivePreviewActivity.class,
                    CameraXSourceDemoActivity.class,
            };

    private static final int[] DESCRIPTION_IDS =
            VERSION.SDK_INT < VERSION_CODES.LOLLIPOP
                    ? new int[]{
                    R.string.desc_camera_source_activity, R.string.desc_still_image_activity,
            }
                    : new int[]{
                    R.string.desc_camera_source_activity,
                    R.string.desc_still_image_activity,
                    R.string.desc_camerax_live_preview_activity,
                    R.string.desc_cameraxsource_demo_activity,
            };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // GET_RUNTIME_PERMISSIONS
        if (!allRuntimePermissionsGranted()) {
            getRuntimePermissions();
        }

        if (BuildConfig.DEBUG) {
            StrictMode.setThreadPolicy(
                    new StrictMode.ThreadPolicy.Builder().detectAll().penaltyLog().build());
            StrictMode.setVmPolicy(
                    new StrictMode.VmPolicy.Builder()
                            .detectLeakedSqlLiteObjects()
                            .detectLeakedClosableObjects()
                            .penaltyLog()
                            .build());
        }
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_chooser);

        // Set up ListView and Adapter
        ListView listView = findViewById(R.id.test_activity_list_view);

        MyArrayAdapter adapter = new MyArrayAdapter(this, android.R.layout.simple_list_item_2, CLASSES);
        adapter.setDescriptionIds(DESCRIPTION_IDS);

        listView.setAdapter(adapter);
        listView.setOnItemClickListener(this);
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


    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        Class<?> clicked = CLASSES[position];
        startActivity(new Intent(this, clicked));
    }

    private static class MyArrayAdapter extends ArrayAdapter<Class<?>> {

        private final Context context;
        private final Class<?>[] classes;
        private int[] descriptionIds;

        MyArrayAdapter(Context context, int resource, Class<?>[] objects) {
            super(context, resource, objects);

            this.context = context;
            classes = objects;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View view = convertView;

            if (convertView == null) {
                LayoutInflater inflater =
                        (LayoutInflater) context.getSystemService(LAYOUT_INFLATER_SERVICE);
                view = inflater.inflate(android.R.layout.simple_list_item_2, null);
            }

            ((TextView) view.findViewById(android.R.id.text1)).setText(classes[position].getSimpleName());
            ((TextView) view.findViewById(android.R.id.text2)).setText(descriptionIds[position]);

            return view;
        }

        void setDescriptionIds(int[] descriptionIds) {
            this.descriptionIds = descriptionIds;
        }
    }
}
