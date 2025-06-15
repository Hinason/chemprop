# -*- coding: utf-8 -*-
# mindspore_chem/train.py

import os
from tqdm import tqdm

# MindSpore
import mindspore
from mindspore import nn, ops, save_checkpoint, load_checkpoint

from data_pre import create_data_loader
from models import MPN
from utils import Metric, plot_evaluation_summary

def run_training(args, train_data, val_data):
    """执行完整的训练和验证流程"""
    train_loader = create_data_loader(train_data, args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_data, args.batch_size, shuffle=False)
    model = MPN(hidden_size=args.hidden_size, depth=args.depth, dropout_prob=args.dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.learning_rate)
    def forward_fn(graph, labels):
        logits = model(graph)
        loss = loss_fn(logits, labels)
        return loss, logits
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    best_val_auc = 0.0
    val_metric_calculator = Metric()
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.ckpt")
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.set_train(True)
        total_loss, step_count = 0, 0
        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_data in progress_bar:
            if batch_data is None: 
                continue
            batch_graph, labels = batch_data
            (loss, _), grads = grad_fn(batch_graph, labels)
            optimizer(grads)
            loss_val = loss.asnumpy()
            total_loss += loss_val
            step_count += 1
            progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
        avg_train_loss = total_loss / step_count if step_count > 0 else 0
        model.set_train(False)
        val_metric_calculator.reset()
        
        for batch_data in val_loader:
            if batch_data is None:
                continue
            batch_graph, labels = batch_data
            preds = model(batch_graph)
            val_metric_calculator.update(preds, labels)
        current_val_auc = val_metric_calculator.compute('AUC')
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val AUC: {current_val_auc:.4f}")
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_checkpoint(model, best_model_path)
            print(f"  -> New best model saved with AUC: {best_val_auc:.4f}")
    print("Training finished.")
    return best_model_path

def run_testing(args, test_data, best_model_path):
    """加载最佳模型并在测试集上进行评估"""
    print("\nStarting final testing...")
    test_loader = create_data_loader(test_data, args.batch_size, shuffle=False)
    model = MPN(hidden_size=args.hidden_size, depth=args.depth, dropout_prob=args.dropout)
    load_checkpoint(best_model_path, net=model)
    model.set_train(False)
    print(f"Loaded best model from: {best_model_path}")
    test_metric_calculator = Metric()
    for batch_data in tqdm(test_loader, desc="Testing"):
        if batch_data is None:
            continue
        batch_graph, labels = batch_data
        preds = model(batch_graph)
        test_metric_calculator.update(preds, labels)
    test_auc = test_metric_calculator.compute('AUC')
    test_acc = test_metric_calculator.compute('Accuracy')
    test_ap = test_metric_calculator.compute('AveragePrecision')
    print("\n===== Final Test Results =====")
    print(f"  Test Set AUC:                 {test_auc:.4f}")
    print(f"  Test Set Accuracy:            {test_acc:.4f}")
    print(f"  Test Set Average Precision:   {test_ap:.4f}") 
    print("============================")
    y_true, y_probs = test_metric_calculator.get_labels_and_probs()
    if y_true and y_probs: 
        plot_title_prefix = args.dataset_name if hasattr(args, 'dataset_name') else "TestSet"
        plot_save_directory = getattr(args, 'plot_save_dir', None)
        if plot_save_directory:
             plot_evaluation_summary(y_true, y_probs, model_name=f"{plot_title_prefix} Performance", save_dir=plot_save_directory)
        else:
            print("Plot save directory not specified in args. Displaying plots.")
            plot_evaluation_summary(y_true, y_probs, model_name=f"{plot_title_prefix} Performance", save_dir=None)
    else:
        print("No data available to plot evaluation curves.")