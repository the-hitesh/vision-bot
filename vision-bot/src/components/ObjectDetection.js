import React, { useRef, useState, useEffect } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

const ObjectDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const detectionIntervalRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsLoading(true);
        
        // Try to set backend, fallback to CPU if WebGL is not available
        let backendSet = false;
        try {
          await tf.setBackend('webgl');
          await tf.ready();
          backendSet = true;
          console.log('Using WebGL backend');
        } catch (e) {
          console.log('WebGL not available, falling back to CPU backend');
        }
        
        if (!backendSet) {
          try {
            await tf.setBackend('cpu');
            await tf.ready();
            console.log('Using CPU backend');
          } catch (e) {
            console.error('Failed to initialize CPU backend:', e);
            throw new Error('No backend available');
          }
        }
        
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        setIsLoading(false);
      } catch (err) {
        console.error('Model loading error:', err);
        setError('Failed to load AI model. Please refresh the page.');
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
        };
      }
    } catch (err) {
      setError('Failed to access camera. Please grant camera permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const detectObjects = async () => {
    if (model && videoRef.current && videoRef.current.readyState === 4) {
      const predictions = await model.detect(videoRef.current);
      setDetections(predictions);
      drawBoundingBoxes(predictions);
    }
  };

  const drawBoundingBoxes = (predictions) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = '#3b82f6';
      ctx.font = '16px Arial';
      const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillRect(x, y - 25, textWidth + 10, 25);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 5, y - 7);
    });
  };

  const handleStartDetection = async () => {
    if (!model) return;
    
    await startCamera();
    setIsDetecting(true);
    
    detectionIntervalRef.current = setInterval(() => {
      detectObjects();
    }, 100);
  };

  const handleStopDetection = () => {
    setIsDetecting(false);
    stopCamera();
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    setDetections([]);
    
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  useEffect(() => {
    return () => {
      handleStopDetection();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <p className="text-xl font-semibold text-gray-700">Loading AI Model...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-red-50 to-pink-100">
        <div className="bg-white p-8 rounded-lg shadow-lg max-w-md">
          <div className="text-red-600 text-5xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Error</h2>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-800 mb-3">
            🤖 AI Vision Bot
          </h1>
          <p className="text-lg text-gray-600">
            Real-time object detection powered by TensorFlow.js
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
                <video
                  ref={videoRef}
                  className="absolute inset-0 w-full h-full object-cover"
                  autoPlay
                  playsInline
                  muted
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full"
                />
                {!isDetecting && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
                    <div className="text-center text-white">
                      <div className="text-6xl mb-4">📹</div>
                      <p className="text-xl font-semibold">Camera Ready</p>
                      <p className="text-sm mt-2 text-gray-300">Click "Start Detection" to begin</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 flex gap-4 justify-center">
                {!isDetecting ? (
                  <button
                    onClick={handleStartDetection}
                    className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-indigo-700 transform hover:scale-105 transition-all duration-200"
                  >
                    ▶️ Start Detection
                  </button>
                ) : (
                  <button
                    onClick={handleStopDetection}
                    className="px-8 py-3 bg-gradient-to-r from-red-600 to-pink-600 text-white font-semibold rounded-lg shadow-lg hover:from-red-700 hover:to-pink-700 transform hover:scale-105 transition-all duration-200"
                  >
                    ⏹️ Stop Detection
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                <span className="mr-2">🎯</span>
                Detected Objects
              </h2>
              
              {detections.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <div className="text-5xl mb-3">👁️</div>
                  <p className="text-sm">No objects detected yet</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {detections.map((detection, index) => (
                    <div
                      key={index}
                      className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200 hover:shadow-md transition-shadow"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold text-gray-800 capitalize">
                          {detection.class}
                        </span>
                        <span className="text-sm font-medium text-blue-600">
                          {Math.round(detection.score * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${detection.score * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {detections.length > 0 && (
                <div className="mt-6 pt-4 border-t border-gray-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-600">Total Objects:</span>
                    <span className="text-2xl font-bold text-blue-600">{detections.length}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white rounded-2xl shadow-xl p-6 mt-6">
              <h3 className="text-lg font-bold text-gray-800 mb-3">ℹ️ About</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                This AI vision bot uses the COCO-SSD model to detect and identify objects in real-time. 
                It can recognize 90 different object classes including people, animals, vehicles, and everyday items.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetection;
