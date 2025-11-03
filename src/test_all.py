#!/usr/bin/env python3
"""
test_all.py - Evaluates the model on all test images
"""
import os
from pathlib import Path
import torch
from tqdm import tqdm
import json
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from detect import load_model, preprocess, predict

def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()

def main():
    # Parameters
    test_dir = Path("/Users/dasaradha/Coding/projects/4-1-min-project/Split Dataset/Test")
    model_dir = Path("experiments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model, classes = load_model(model_dir, device)
    
    # Collect all test images
    all_predictions = []
    all_true_labels = []
    confidence_scores = defaultdict(list)
    results_by_class = defaultdict(lambda: defaultdict(int))
    
    print("\nProcessing test images...")
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        true_class = class_dir.name
        print(f"\nTesting {true_class}...")
        
        for img_path in tqdm(list(class_dir.glob('*.jpg'))):
            # Preprocess and predict
            tensor = preprocess(str(img_path))
            pred_idx, confidence = predict(model, tensor, device)
            pred_class = classes[pred_idx]
            
            # Store results
            all_predictions.append(pred_class)
            all_true_labels.append(true_class)
            confidence_scores[true_class].append(confidence)
            
            # Update per-class statistics
            results_by_class[true_class]['total'] += 1
            if pred_class == true_class:
                results_by_class[true_class]['correct'] += 1

    # Calculate and display results
    print("\n=== Test Results ===")
    
    # Per-class accuracy
    print("\nPer-class Performance:")
    for class_name in sorted(results_by_class.keys()):
        total = results_by_class[class_name]['total']
        correct = results_by_class[class_name]['correct']
        accuracy = (correct / total) * 100
        avg_confidence = np.mean(confidence_scores[class_name]) * 100
        print(f"{class_name:15s}: {accuracy:5.1f}% accuracy ({correct}/{total}), "
              f"avg confidence: {avg_confidence:.1f}%")

    # Overall metrics
    print("\nDetailed Classification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_path = model_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(cm, classes, cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    # Save detailed results
    results = {
        'overall_accuracy': (sum(1 for x, y in zip(all_predictions, all_true_labels) 
                               if x == y) / len(all_predictions)) * 100,
        'per_class_accuracy': {
            class_name: results_by_class[class_name]['correct'] / results_by_class[class_name]['total'] * 100
            for class_name in results_by_class
        },
        'per_class_confidence': {
            class_name: float(np.mean(scores)) * 100
            for class_name, scores in confidence_scores.items()
        }
    }
    
    with open(model_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {model_dir}/test_results.json")

if __name__ == "__main__":
    main()