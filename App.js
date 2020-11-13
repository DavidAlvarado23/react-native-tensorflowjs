import React, { useState, useEffect } from "react";
import {
  View,
  Button,
  Text,
  ScrollView,
  Image,
  PermissionsAndroid,
  StyleSheet,
} from "react-native";

import { Camera } from "expo-camera";
import * as FileSystem from "expo-file-system";

import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
import { decodeJpeg, bundleResourceIO } from "@tensorflow/tfjs-react-native";

import ImagePicker from "react-native-image-picker";

// To load local models
// Get reference to bundled model assets
import modelJson from "./src/assets/model/model.json";
import modelWeights from "./src/assets/model/weights.bin";

const App = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [isTfReady, setIsTfReady] = useState(false);
  const [image, setImage] = React.useState(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState([]);

  const activateTensorflow = async () => {
    // Wait for tf to be ready.
    await tf.ready();
    // Signal to the app that tensorflow.js can now be used.
    setIsTfReady(true);
    loadLocalModel();
  };

  const requestPermissions = async () => {
    const { status } = await Camera.requestPermissionsAsync();
    setHasPermission(status === "granted");
  };

  const loadModel = async () => {
    const model = await mobilenet.load();
    setModel(model);
  };

  const loadLocalModel = async () => {
    // Use the bundleResorceIO IOHandler to load the model
    const model = await tf.loadLayersModel(
      bundleResourceIO(modelJson, modelWeights)
    );

    const feature = tf.ones([1, 8]);
    const y = model.predict(feature);
    const response = y.arraySync();
    console.log(response[0]);
  };

  useEffect(() => {
    activateTensorflow();
    requestPermissions();
    loadModel();
  }, []);

  const takePicture = async () => {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE
    );

    if (granted === PermissionsAndroid.RESULTS.GRANTED) {
      ImagePicker.launchCamera(
        {
          mediaType: "photo",
          includeBase64: false,
          maxHeight: 200,
          maxWidth: 200,
        },
        (response) => {
          setImage(response);
        }
      );
    }
  };

  const loadImageAndPredict = async () => {
    try {
      // Read the image from the local file system encoded in base64.
      const imgB64 = await FileSystem.readAsStringAsync(image.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      // Get the image as a buffer.
      const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
      const raw = new Uint8Array(imgBuffer);
      // Transform the image to a valid tensor.
      const imageTensor = decodeJpeg(raw);

      // Send the image tensor to the model and get a prediction.
      const prediction = await model.classify(imageTensor);
      setPrediction(prediction);
    } catch (err) {
      console.error(err);
    }
  };

  const reset = () => {
    setImage(null);
    setPrediction([]);
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <View style={{ flex: 1, justifyContent: "flex-end", marginBottom: 20 }}>
        <Button title="Take image" onPress={takePicture} />
        <View style={{ marginTop: 20 }}>
          <Button
            title="Predict"
            disabled={!isTfReady || !model || !image}
            onPress={loadImageAndPredict}
          />
        </View>
        <View style={{ marginTop: 20 }}>
          <Button
            title="Reset"
            disabled={!isTfReady || !model || !image || !prediction}
            onPress={reset}
          />
        </View>
      </View>
      {image && (
        <View style={{ alignItems: "center", margin: 20 }}>
          <Image
            style={{ width: 200, height: 200 }}
            source={{ uri: image.uri }}
          />
        </View>
      )}
      <ScrollView style={{ flex: 1 }}>
        <View style={styles.section}>
          <Text style={{ fontWeight: "bold" }}>Classname</Text>
          <Text style={{ fontWeight: "bold" }}>Probability</Text>
        </View>
        {prediction.map((a) => (
          <View style={styles.section}>
            <Text>{a.className}</Text>
            <Text>{a.probability}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  section: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "space-between",
  },
});

export default App;
