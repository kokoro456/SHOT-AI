plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.shot.detection"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Don't compress TFLite model files
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation(project(":core"))

    // TensorFlow Lite
    implementation(libs.tflite)
    implementation(libs.tflite.gpu)
    implementation(libs.tflite.support)

    // Coroutines
    implementation(libs.coroutines.core)

    // AndroidX
    implementation(libs.androidx.core)

    testImplementation(libs.junit)
}
