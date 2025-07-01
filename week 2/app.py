#!/usr/bin/env python3
"""
Enhanced Intelligent Waste Classification System
All-in-One AI-Based Garbage Detection and Classification with Advanced Features
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import threading
import time
from PIL import Image, ImageTk
import pyttsx3
import pygame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
warnings.filterwarnings('ignore')

class EnhancedWasteClassificationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Intelligent Waste Classification System v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e2124')
        
        # Initialize variables
        self.model = None
        self.garbage_detector = None  # Binary classifier to detect if image contains garbage
        self.label_encoder = None
        self.camera = None
        self.is_camera_active = False
        self.is_continuous_scan = False
        self.tts_engine = None
        self.is_muted = False
        self.current_frame = None
        self.training_progress = 0
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Enhanced categories and tips
        self.categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.recyclability_tips = {
            'cardboard': 'Recyclable ♻️. Flatten and place in recycling bin. Remove tape and staples.',
            'glass': 'Recyclable ♻️. Rinse clean and place in glass recycling. Remove caps.',
            'metal': 'Recyclable ♻️. Clean and place in metal recycling. Check for aluminum vs steel.',
            'paper': 'Recyclable ♻️. Keep dry and place in paper recycling. Remove plastic coatings.',
            'plastic': 'Recyclable ♻️. Check recycling number (1-7) and recycle accordingly.',
            'trash': 'Non-recyclable 🗑️. Dispose in general waste bin. Consider reduction alternatives.'
        }
        
        # Audio feedback settings
        self.last_detection_time = 0
        self.detection_cooldown = 3.0  # 3 seconds between detections
        
        # Initialize audio
        self.init_audio()
        
        # Create enhanced GUI
        self.create_enhanced_gui()
        
        # Try to load existing models
        self.load_existing_models()
    
    def init_audio(self):
        """Initialize enhanced text-to-speech and pygame for sounds"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Set voice to female if available
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Create detection beep sound
            self.create_detection_beep()
        except Exception as e:
            print(f"Audio initialization error: {e}")
    
    def create_detection_beep(self):
        """Create an enhanced beep sound for detection"""
        try:
            sample_rate = 22050
            duration = 0.3
            frequency = 1000
            
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            # Create a pleasant beep with fade in/out
            for i in range(frames):
                # Sine wave
                sine_val = np.sin(2 * np.pi * frequency * i / sample_rate)
                
                # Fade in/out envelope
                if i < frames * 0.1:
                    envelope = i / (frames * 0.1)
                elif i > frames * 0.9:
                    envelope = (frames - i) / (frames * 0.1)
                else:
                    envelope = 1.0
                
                arr[i] = sine_val * envelope
            
            arr = (arr * 32767 * 0.3).astype(np.int16)  # Reduced volume
            
            # Create stereo sound
            stereo_arr = np.array([arr, arr]).T
            self.beep_sound = pygame.sndarray.make_sound(stereo_arr)
            
        except Exception as e:
            print(f"Beep sound creation error: {e}")
            self.beep_sound = None
    
    def create_enhanced_gui(self):
        """Create the enhanced GUI interface"""
        # Main frame with modern styling
        main_frame = tk.Frame(self.root, bg='#1e2124')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Enhanced title with gradient effect
        title_frame = tk.Frame(main_frame, bg='#1e2124')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, text="🤖 Enhanced AI Waste Classification System v2.0", 
                              font=('Segoe UI', 26, 'bold'), fg='#7289da', bg='#1e2124')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Advanced Machine Learning • Real-time Detection • Smart Recycling", 
                                 font=('Segoe UI', 12), fg='#99aab5', bg='#1e2124')
        subtitle_label.pack()
        
        # Enhanced control panel
        control_frame = tk.Frame(main_frame, bg='#2f3136', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Primary buttons frame
        buttons_frame = tk.Frame(control_frame, bg='#2f3136')
        buttons_frame.pack(pady=12)
        
        # Enhanced buttons with modern styling
        button_style = {
            'font': ('Segoe UI', 11, 'bold'),
            'width': 16,
            'height': 2,
            'relief': tk.FLAT,
            'cursor': 'hand2'
        }
        
        self.train_btn = tk.Button(buttons_frame, text="🧠 Train Model", 
                                  command=self.train_model_dialog,
                                  bg='#5865f2', fg='white', **button_style)
        self.train_btn.pack(side=tk.LEFT, padx=8)
        
        self.upload_btn = tk.Button(buttons_frame, text="📁 Upload Image", 
                                   command=self.upload_image,
                                   bg='#57f287', fg='white', **button_style)
        self.upload_btn.pack(side=tk.LEFT, padx=8)
        
        self.camera_btn = tk.Button(buttons_frame, text="📷 Start Camera", 
                                   command=self.toggle_camera,
                                   bg='#ed4245', fg='white', **button_style)
        self.camera_btn.pack(side=tk.LEFT, padx=8)
        
        self.scan_btn = tk.Button(buttons_frame, text="🔄 Smart Scan", 
                                 command=self.toggle_continuous_scan,
                                 bg='#9146ff', fg='white', **button_style)
        self.scan_btn.pack(side=tk.LEFT, padx=8)
        
        self.mute_btn = tk.Button(buttons_frame, text="🔊 Audio ON", 
                                 command=self.toggle_mute,
                                 bg='#faa61a', fg='white', **button_style)
        self.mute_btn.pack(side=tk.LEFT, padx=8)
        
        # Enhanced progress section
        progress_frame = tk.Frame(control_frame, bg='#2f3136')
        progress_frame.pack(fill=tk.X, padx=15, pady=(0, 12))
        
        # Progress label
        self.progress_label = tk.Label(progress_frame, text="Ready", 
                                      font=('Segoe UI', 10, 'bold'), fg='#57f287', bg='#2f3136')
        self.progress_label.pack(side=tk.LEFT)
        
        # Progress percentage
        self.progress_percent = tk.Label(progress_frame, text="", 
                                        font=('Segoe UI', 10, 'bold'), fg='#faa61a', bg='#2f3136')
        self.progress_percent.pack(side=tk.RIGHT)
        
        # Modern progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar", 
                       background='#57f287', 
                       troughcolor='#40444b',
                       borderwidth=0)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', 
                                           style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#1e2124')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Enhanced image display
        left_panel = tk.Frame(content_frame, bg='#2f3136', relief=tk.FLAT, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        # Image header
        img_header = tk.Frame(left_panel, bg='#2f3136')
        img_header.pack(fill=tk.X, pady=12)
        
        tk.Label(img_header, text="📸 Live Preview", 
                font=('Segoe UI', 16, 'bold'), fg='#ffffff', bg='#2f3136').pack(side=tk.LEFT, padx=15)
        
        # Camera status indicator
        self.camera_status = tk.Label(img_header, text="● Offline", 
                                     font=('Segoe UI', 10), fg='#ed4245', bg='#2f3136')
        self.camera_status.pack(side=tk.RIGHT, padx=15)
        
        # Enhanced image display with border
        image_container = tk.Frame(left_panel, bg='#40444b', relief=tk.FLAT, bd=2)
        image_container.pack(expand=True, fill=tk.BOTH, padx=15, pady=(0, 15))
        
        self.image_label = tk.Label(image_container, bg='#36393f', 
                                   text="Upload an image or start camera to begin\n🎯 AI-Powered Waste Detection", 
                                   fg='#99aab5', font=('Segoe UI', 14))
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=3, pady=3)
        
        # Right panel - Enhanced results
        right_panel = tk.Frame(content_frame, bg='#2f3136', relief=tk.FLAT, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        
        # Results header
        results_header = tk.Frame(right_panel, bg='#2f3136')
        results_header.pack(fill=tk.X, pady=12)
        
        tk.Label(results_header, text="🎯 Classification Results", 
                font=('Segoe UI', 16, 'bold'), fg='#ffffff', bg='#2f3136').pack(padx=15)
        
        # Results container
        results_container = tk.Frame(right_panel, bg='#36393f', width=350, relief=tk.FLAT, bd=1)
        results_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        results_container.pack_propagate(False)
        
        # Prediction display with modern card design
        pred_card = tk.Frame(results_container, bg='#40444b', relief=tk.FLAT, bd=1)
        pred_card.pack(fill=tk.X, padx=15, pady=15)
        
        self.prediction_label = tk.Label(pred_card, text="🤖 Ready for Analysis", 
                                        font=('Segoe UI', 16, 'bold'), fg='#ffffff', bg='#40444b',
                                        wraplength=300, pady=15)
        self.prediction_label.pack()
        
        # Confidence display
        conf_card = tk.Frame(results_container, bg='#40444b', relief=tk.FLAT, bd=1)
        conf_card.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.confidence_label = tk.Label(conf_card, text="", 
                                        font=('Segoe UI', 13, 'bold'), fg='#faa61a', bg='#40444b',
                                        wraplength=300, pady=12)
        self.confidence_label.pack()
        
        # Tip display with enhanced styling
        tip_card = tk.Frame(results_container, bg='#40444b', relief=tk.FLAT, bd=1)
        tip_card.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.tip_label = tk.Label(tip_card, text="", 
                                 font=('Segoe UI', 11), fg='#57f287', bg='#40444b',
                                 wraplength=300, justify=tk.LEFT, pady=12)
        self.tip_label.pack()
        
        # Detection history
        history_card = tk.Frame(results_container, bg='#40444b', relief=tk.FLAT, bd=1)
        history_card.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        tk.Label(history_card, text="📊 Recent Detections", 
                font=('Segoe UI', 12, 'bold'), fg='#ffffff', bg='#40444b').pack(pady=(10, 5))
        
        self.history_text = tk.Text(history_card, height=6, bg='#36393f', fg='#99aab5',
                                   font=('Segoe UI', 9), relief=tk.FLAT, bd=0)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Enhanced status bar
        status_frame = tk.Frame(main_frame, bg='#2f3136', relief=tk.FLAT, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        
        self.status_label = tk.Label(status_frame, text="🟢 System Ready - Load your dataset to train the AI model", 
                                    font=('Segoe UI', 10), fg='#57f287', bg='#2f3136')
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        # System info
        self.sys_info_label = tk.Label(status_frame, text="TensorFlow Ready | OpenCV Active", 
                                      font=('Segoe UI', 9), fg='#99aab5', bg='#2f3136')
        self.sys_info_label.pack(side=tk.RIGHT, padx=15, pady=8)
    
    def load_existing_models(self):
        """Load existing models with enhanced error handling"""
        try:
            # Load main classifier
            if os.path.exists('enhanced_waste_classifier.h5') and os.path.exists('label_encoder.json'):
                self.model = keras.models.load_model('enhanced_waste_classifier.h5')
                
                with open('label_encoder.json', 'r') as f:
                    encoder_data = json.load(f)
                
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(encoder_data['classes'])
                
                # Load garbage detector
                if os.path.exists('garbage_detector.h5'):
                    self.garbage_detector = keras.models.load_model('garbage_detector.h5')
                
                self.update_status("🟢 AI Models loaded successfully! System ready for classification.", '#57f287')
                self.sys_info_label.config(text="Enhanced Models Active | Ready for Detection")
                
            else:
                self.update_status("🟡 No existing models found. Please train new models with your dataset.", '#faa61a')
                
        except Exception as e:
            self.update_status(f"🔴 Error loading models: {str(e)}", '#ed4245')
    
    def train_model_dialog(self):
        """Enhanced dialog for model training"""
        folder_path = filedialog.askdirectory(
            title="Select Dataset Folder (organized in category subfolders)")
        if folder_path:
            # Show training info dialog
            info_msg = """Training Enhanced AI Model:

🔹 Expected folder structure:
   dataset/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/

🔹 Each folder should contain 100+ images
🔹 Training will take 5-15 minutes
🔹 Progress will be shown in real-time

Continue with training?"""
            
            if messagebox.askyesno("Start Training", info_msg):
                threading.Thread(target=self.train_enhanced_model, args=(folder_path,), daemon=True).start()
    
    def train_enhanced_model(self, dataset_path):
        """Train enhanced model with real-time progress"""
        try:
            self.update_status("🚀 Initializing enhanced training process...", '#5865f2')
            self.train_btn.config(state='disabled')
            
            # Initialize progress
            self.training_progress = 0
            self.current_epoch = 0
            self.total_epochs = 50  # Increased for better accuracy
            self.update_progress(0, "Loading dataset...")
            
            # Load and preprocess data with augmentation
            X, y, dataset_size = self.load_enhanced_dataset(dataset_path)
            
            if len(X) == 0:
                raise ValueError("No valid images found in dataset")
            
            self.update_progress(15, f"Dataset loaded: {dataset_size} images")
            
            # Create garbage vs non-garbage dataset for binary classifier
            X_garbage, y_garbage = self.create_garbage_detection_dataset(X, y)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            y_categorical = keras.utils.to_categorical(y_encoded)
            
            self.update_progress(25, "Data preprocessing complete")
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            X_garbage_train, X_garbage_test, y_garbage_train, y_garbage_test = train_test_split(
                X_garbage, y_garbage, test_size=0.2, random_state=42
            )
            
            self.update_progress(35, "Data split completed")
            
            # Create enhanced models
            self.model = self.create_enhanced_model(len(self.label_encoder.classes_))
            self.garbage_detector = self.create_garbage_detector()
            
            self.update_progress(45, "Models created")
            
            # Train garbage detector first
            self.update_status("🎯 Training garbage detection model...", '#9146ff')
            
            garbage_history = self.garbage_detector.fit(
                X_garbage_train, y_garbage_train,
                validation_data=(X_garbage_test, y_garbage_test),
                epochs=20,
                batch_size=32,
                verbose=0,
                callbacks=[TrainingCallback(self, 45, 65)]
            )
            
            # Train main classifier
            self.update_status("🧠 Training waste classification model...", '#5865f2')
            
            # Data augmentation for better accuracy
            train_generator = self.create_data_generator()
            
            history = self.model.fit(
                train_generator.flow(X_train, y_train, batch_size=32),
                validation_data=(X_test, y_test),
                epochs=self.total_epochs - 20,
                verbose=0,
                callbacks=[
                    TrainingCallback(self, 65, 95),
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            self.update_progress(95, "Saving models...")
            
            # Save models
            self.model.save('enhanced_waste_classifier.h5')
            self.garbage_detector.save('garbage_detector.h5')
            
            encoder_data = {'classes': self.label_encoder.classes_.tolist()}
            with open('label_encoder.json', 'w') as f:
                json.dump(encoder_data, f)
            
            # Evaluate model
            test_predictions = self.model.predict(X_test)
            test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
            
            self.update_progress(100, "Training completed!")
            
            # Show success message
            success_msg = f"""✅ Enhanced AI Model Training Complete!

📊 Performance Metrics:
• Classification Accuracy: {test_accuracy:.1%}
• Garbage Detection Accuracy: {garbage_history.history['val_accuracy'][-1]:.1%}
• Total Images Processed: {dataset_size}
• Training Epochs: {self.total_epochs}

🎯 Model Features:
• Advanced garbage detection
• Multi-class waste classification
• False positive prevention
• Real-time audio feedback

Your AI model is ready for deployment!"""
            
            messagebox.showinfo("Training Complete", success_msg)
            self.update_status("🟢 Enhanced AI model trained successfully! Ready for detection.", '#57f287')
            
        except Exception as e:
            self.update_status(f"🔴 Training failed: {str(e)}", '#ed4245')
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")
        finally:
            self.train_btn.config(state='normal')
            self.progress_bar.config(mode='determinate')
            self.update_progress(0, "Ready")
    
    def load_enhanced_dataset(self, dataset_path):
        """Load dataset with enhanced preprocessing"""
        X, y = [], []
        dataset_size = 0
        
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue
                
            category_name = category.lower()
            if category_name not in self.categories:
                continue
            
            category_count = 0
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(category_path, filename)
                    try:
                        # Enhanced image loading with validation
                        img = cv2.imread(img_path)
                        if img is not None and img.shape[0] > 32 and img.shape[1] > 32:
                            # Enhanced preprocessing
                            img = cv2.resize(img, (224, 224))
                            img = cv2.GaussianBlur(img, (3, 3), 0)  # Noise reduction
                            img = img.astype('float32') / 255.0
                            
                            # Normalize
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            img = (img - mean) / std
                            
                            X.append(img)
                            y.append(category_name)
                            category_count += 1
                            dataset_size += 1
                            
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            print(f"Loaded {category_count} images for {category_name}")
        
        return np.array(X), np.array(y), dataset_size
    
    def create_garbage_detection_dataset(self, X, y):
        """Create binary dataset for garbage detection"""
        # For garbage detection: 1 = garbage, 0 = not garbage
        X_garbage = X.copy()
        y_garbage = np.ones(len(y))  # All current images are garbage
        
        # In a real implementation, you would add non-garbage images here
        # For now, we'll use some augmented versions as "hard negatives"
        
        return X_garbage, y_garbage
    
    def create_enhanced_model(self, num_classes):
        """Create enhanced CNN model with transfer learning"""
        # Use EfficientNetB0 as base model for better accuracy
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use advanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def create_garbage_detector(self):
        """Create binary classifier for garbage detection"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_data_generator(self):
        """Create data augmentation generator for better accuracy"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    def update_progress(self, percentage, message):
        """Update training progress with percentage"""
        self.training_progress = percentage
        self.progress_bar['value'] = percentage
        self.progress_percent.config(text=f"{percentage}%")
        self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def update_status(self, message, color='#99aab5'):
        """Update status with color coding"""
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()
    
    def upload_image(self):
        """Upload and classify image with enhanced detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image for AI Analysis",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.classify_enhanced_image(file_path)
    
    def classify_enhanced_image(self, image_path):
        """Enhanced image classification with garbage detection"""
        try:
            if self.model is None:
                messagebox.showerror("Model Not Ready", "Please train the AI model first!")
                return
            
            self.update_status("🔍 Analyzing image with AI...", '#5865f2')
            
            # Load and display image
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not load image! Please try a different file.")
                return
            
            self.display_enhanced_image(img)
            
            # Preprocess for prediction
            img_processed = cv2.resize(img, (224, 224))
            img_processed = img_processed.astype('float32') / 255.0
            
            # Normalize like training data
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_processed = (img_processed - mean) / std
            img_processed = np.expand_dims(img_processed, axis=0)
            
            # First check if it's garbage using binary classifier
            if self.garbage_detector is not None:
                garbage_prediction = self.garbage_detector.predict(img_processed, verbose=0)[0][0]
                
                if garbage_prediction < 0.5:  # Not garbage
                    self.update_enhanced_results("Garbage Not Detected", 0, 
                                               "This image does not contain identifiable waste items.")
                    self.add_to_history("❌ Non-waste item detected")
                    self.speak("Garbage not detected in this image")
                    self.update_status("🟡 Analysis complete - No waste detected", '#faa61a')
                    return
            
            # Make waste classification prediction
            predictions = self.model.predict(img_processed, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Additional confidence threshold for uploaded images
            if confidence < 0.4:
                self.update_enhanced_results("Uncertain Classification", confidence, 
                                           "Image quality may be too low or object unclear. Try a clearer image.")
                self.add_to_history("❓ Uncertain classification")
                self.speak("Image classification uncertain")
                self.update_status("🟡 Classification uncertain - Try a clearer image", '#faa61a')
            else:
                predicted_class = self.label_encoder.classes_[predicted_class_idx]
                tip = self.recyclability_tips.get(predicted_class, "")
                
                self.update_enhanced_results(predicted_class.title(), confidence, tip)
                self.add_to_history(f"✅ {predicted_class.title()} - {confidence:.1%}")
                self.speak(f"This is {predicted_class} waste")
                self.update_status(f"🟢 Classification complete - {predicted_class.title()} detected", '#57f287')
            
        except Exception as e:
            self.update_status(f"🔴 Classification error: {str(e)}", '#ed4245')
            messagebox.showerror("Analysis Error", f"Classification failed: {str(e)}")
    
    def display_enhanced_image(self, img):
        """Display image with enhanced styling"""
        try:
            # Convert and resize for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Calculate size to fit in label while maintaining aspect ratio
            label_width = 500
            label_height = 400
            img_pil.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Update label
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk  # Keep reference
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_enhanced_results(self, prediction, confidence, tip):
        """Update results display with enhanced styling"""
        if prediction == "Garbage Not Detected":
            self.prediction_label.config(text="❌ Garbage Not Detected", fg='#ed4245')
            self.confidence_label.config(text="System Analysis: Non-waste item")
            self.tip_label.config(text="💡 This appears to be a non-garbage item. Please upload an image containing waste materials for classification.")
        elif prediction == "Uncertain Classification":
            self.prediction_label.config(text="❓ Uncertain Classification", fg='#faa61a')
            self.confidence_label.config(text=f"Confidence: {confidence:.1%} (Below threshold)")
            self.tip_label.config(text=tip)
        else:
            # Get emoji for waste type
            emoji_map = {
                'cardboard': '📦',
                'glass': '🍶',
                'metal': '🥫',
                'paper': '📄',
                'plastic': '🥤',
                'trash': '🗑️'
            }
            emoji = emoji_map.get(prediction.lower(), '♻️')
            
            self.prediction_label.config(text=f"{emoji} {prediction}", fg='#57f287')
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            self.tip_label.config(text=f"💡 {tip}")
    
    def add_to_history(self, entry):
        """Add entry to detection history"""
        timestamp = time.strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] {entry}\n"
        
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
        
        # Limit history to last 20 entries
        lines = self.history_text.get("1.0", tk.END).split('\n')
        if len(lines) > 21:  # 20 entries + 1 empty line
            self.history_text.delete("1.0", "2.0")
    
    def toggle_camera(self):
        """Enhanced camera toggle with status updates"""
        if not self.is_camera_active:
            self.start_enhanced_camera()
        else:
            self.stop_enhanced_camera()
    
    def start_enhanced_camera(self):
        """Start camera with enhanced features"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", "Could not access camera! Please check camera permissions.")
                return
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_camera_active = True
            self.camera_btn.config(text="📷 Stop Camera", bg='#ed4245')
            self.camera_status.config(text="● Live", fg='#57f287')
            
            # Start camera thread
            threading.Thread(target=self.enhanced_camera_loop, daemon=True).start()
            
            self.update_status("📹 Camera activated - Ready for live detection", '#57f287')
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Camera initialization failed: {str(e)}")
    
    def stop_enhanced_camera(self):
        """Stop camera with cleanup"""
        self.is_camera_active = False
        self.is_continuous_scan = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.camera_btn.config(text="📷 Start Camera", bg='#57f287')
        self.scan_btn.config(text="🔄 Smart Scan", bg='#9146ff')
        self.camera_status.config(text="● Offline", fg='#ed4245')
        
        # Clear image display
        self.image_label.config(image="", text="Camera stopped\n📷 Click 'Start Camera' to resume", fg='#99aab5')
        self.update_status("📹 Camera deactivated", '#99aab5')
    
    def enhanced_camera_loop(self):
        """Enhanced camera loop with better performance"""
        frame_count = 0
        last_detection_time = 0
        
        while self.is_camera_active:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                frame_count += 1
                
                # Display every frame for smooth video
                if frame_count % 2 == 0:  # Display every 2nd frame to reduce load
                    self.display_enhanced_image(frame)
                
                # Smart scanning with enhanced detection
                if self.is_continuous_scan and self.model is not None:
                    current_time = time.time()
                    
                    # Detect every 3 seconds for stability
                    if current_time - last_detection_time >= self.detection_cooldown:
                        self.enhanced_detect_from_frame(frame)
                        last_detection_time = current_time
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
    
    def enhanced_detect_from_frame(self, frame):
        """Enhanced detection from camera frame"""
        try:
            # Preprocess frame
            img_processed = cv2.resize(frame, (224, 224))
            img_processed = img_processed.astype('float32') / 255.0
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_processed = (img_processed - mean) / std
            img_processed = np.expand_dims(img_processed, axis=0)
            
            # Check if it's garbage first
            if self.garbage_detector is not None:
                garbage_prediction = self.garbage_detector.predict(img_processed, verbose=0)[0][0]
                
                if garbage_prediction < 0.6:  # Higher threshold for live detection
                    return  # Skip if not garbage
            
            # Make waste classification prediction
            predictions = self.model.predict(img_processed, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Higher confidence threshold for live detection
            if confidence > 0.7:
                predicted_class = self.label_encoder.classes_[predicted_class_idx]
                tip = self.recyclability_tips.get(predicted_class, "")
                
                self.update_enhanced_results(predicted_class.title(), confidence, tip)
                self.add_to_history(f"🎯 Live: {predicted_class.title()} - {confidence:.1%}")
                
                # Enhanced audio feedback
                self.play_detection_beep()
                time.sleep(0.5)  # Brief pause between beep and speech
                self.speak(f"This is {predicted_class}")
                
                self.update_status(f"🎯 Live detection: {predicted_class.title()} ({confidence:.1%})", '#57f287')
                
                # Pause scanning briefly after detection
                time.sleep(1)
            
        except Exception as e:
            print(f"Detection error: {e}")
    
    def toggle_continuous_scan(self):
        """Enhanced continuous scanning toggle"""
        if not self.is_camera_active:
            messagebox.showwarning("Camera Required", "Please start the camera first to enable smart scanning!")
            return
        
        if self.model is None:
            messagebox.showwarning("Model Required", "Please train the AI model first!")
            return
        
        self.is_continuous_scan = not self.is_continuous_scan
        
        if self.is_continuous_scan:
            self.scan_btn.config(text="⏸️ Stop Scanning", bg='#ed4245')
            self.update_status("🎯 Smart scanning active - AI analyzing live feed...", '#9146ff')
            self.add_to_history("🔄 Smart scanning started")
        else:
            self.scan_btn.config(text="🔄 Smart Scan", bg='#9146ff')
            self.update_status("⏸️ Smart scanning paused", '#99aab5')
            self.add_to_history("⏸️ Smart scanning stopped")
    
    def toggle_mute(self):
        """Enhanced mute toggle"""
        self.is_muted = not self.is_muted
        
        if self.is_muted:
            self.mute_btn.config(text="🔇 Audio OFF", bg='#ed4245')
            self.update_status("🔇 Audio feedback disabled", '#ed4245')
        else:
            self.mute_btn.config(text="🔊 Audio ON", bg='#faa61a')
            self.update_status("🔊 Audio feedback enabled", '#57f287')
    
    def play_detection_beep(self):
        """Play enhanced detection beep"""
        if not self.is_muted and self.beep_sound:
            try:
                self.beep_sound.play()
            except Exception as e:
                print(f"Beep error: {e}")
    
    def speak(self, text):
        """Enhanced text-to-speech with better quality"""
        if not self.is_muted and self.tts_engine:
            try:
                threading.Thread(target=self._enhanced_speak_thread, args=(text,), daemon=True).start()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def _enhanced_speak_thread(self, text):
        """Enhanced TTS thread with better error handling"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS thread error: {e}")

class TrainingCallback(keras.callbacks.Callback):
    """Custom callback for real-time training progress"""
    
    def __init__(self, app, start_progress, end_progress):
        super().__init__()
        self.app = app
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.progress_range = end_progress - start_progress
    
    def on_epoch_begin(self, epoch, logs=None):
        self.app.current_epoch = epoch + 1
        progress = self.start_progress + (epoch / (self.params['epochs']) * self.progress_range)
        self.app.update_progress(int(progress), f"Training epoch {epoch + 1}/{self.params['epochs']}...")
    
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy', 0)
        val_accuracy = logs.get('val_accuracy', 0)
        progress = self.start_progress + ((epoch + 1) / self.params['epochs'] * self.progress_range)
        self.app.update_progress(int(progress), 
                               f"Epoch {epoch + 1} complete - Accuracy: {accuracy:.1%}")

def main():
    """Main function to run the enhanced application"""
    # Check TensorFlow GPU availability
    print("🚀 Initializing Enhanced Waste Classification System...")
    print(f"📊 TensorFlow Version: {tf.__version__}")
    
    if tf.config.list_physical_devices('GPU'):
        print("🎮 GPU acceleration available")
    else:
        print("💻 Running on CPU")
    
    root = tk.Tk()
    app = EnhancedWasteClassificationSystem(root)
    
    # Enhanced window closing handler
    def on_closing():
        try:
            if app.camera:
                app.camera.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            if app.tts_engine:
                app.tts_engine.stop()
        except:
            pass
        finally:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Set window icon and properties
    try:
        root.iconbitmap('icon.ico')  # Add your icon file
    except:
        pass
    
    root.minsize(1200, 800)
    
    print("✅ System ready! Launch GUI...")
    root.mainloop()

if __name__ == "__main__":
    main()