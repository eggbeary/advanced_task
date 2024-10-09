# G1 Adaptive Policy 技术改进报告

## 1. 概述

本文档详细描述了G1垃圾收集器策略的技术改进。新的G1AdaptivePolicy在原有G1Policy的基础上引入了自适应机制，利用机器学习技术动态优化GC参数，旨在提高GC效率和应用程序性能。

## 2. 技术改进详情

### 2.1 自适应线性回归模型

#### 2.1.1 AdaptiveLinearRegression类

```cpp
class AdaptiveLinearRegression {
private:
    std::vector<double> _weights;
    std::vector<double> _m;
    std::vector<double> _v;
    double _beta1, _beta2, _epsilon, _alpha;
    int _t;

public:
    AdaptiveLinearRegression(size_t features, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    double predict(const std::vector<double>& features) const;
    void update(const std::vector<double>& features, double target);
    const std::vector<double>& get_weights() const;
};
```

- 实现了Adam优化算法的变体
- 支持在线学习，能够随时间适应变化的数据模式
- 通过调整学习率`alpha`和动量参数`beta1`、`beta2`来控制模型的收敛速度和稳定性

### 2.2 自适应策略核心

#### 2.2.1 G1AdaptivePolicy类

```cpp
class G1AdaptivePolicy : public CHeapObj<mtGC> {
private:
    // 核心模型
    AdaptiveLinearRegression _mixed_gc_live_threshold_model;
    AdaptiveLinearRegression _mixed_gc_count_target_model;
    AdaptiveLinearRegression _old_cset_region_threshold_model;
    AdaptiveLinearRegression _heap_waste_percent_model;

    // 特征工程
    static constexpr size_t HISTORY_LENGTH = 20;
    static constexpr size_t FEATURE_COUNT = 20;
    std::deque<std::vector<double>> _feature_history;

    // 特征选择
    std::vector<bool> _selected_features;
    static constexpr size_t FEATURE_SELECTION_INTERVAL = 20;

    // 方法
    std::vector<double> extract_raw_features();
    std::vector<double> engineer_features(const std::vector<double>& raw_features);
    void update_feature_history(const std::vector<double>& features);
    void perform_feature_selection();
    void parallel_update_models(const std::vector<double>& features);

public:
    void update_after_gc(const G1GCPhaseTimes& phase_times);
    // 其他公共方法...
};
```

### 2.3 特征工程详解

1. **原始特征提取**:
   - 堆使用率
   - 预测的分配率
   - 预测的重新标记时间
   - 预测的清理时间
   - 可回收字节数
   - 最近平均暂停时间比率
   - 当前GC参数值
   - GC效率指标

2. **特征工程**:
   - 指数移动平均值计算
   - 关键指标变化率计算

3. **特征选择**:
   - 基于特征重要性的周期性选择
   - 使用模型权重绝对值作为重要性指标

### 2.4 并行处理机制

```cpp
void G1AdaptivePolicy::parallel_update_models(const std::vector<double>& features) {
    std::vector<std::thread> threads;
    threads.emplace_back([&]() { update_parameter(_mixed_gc_live_threshold_model, _current_mixed_gc_live_threshold, 
                                                  MIN_MIXED_GC_LIVE_THRESHOLD, MAX_MIXED_GC_LIVE_THRESHOLD, features); });
    threads.emplace_back([&]() { update_parameter(_mixed_gc_count_target_model, _current_mixed_gc_count_target, 
                                                  MIN_MIXED_GC_COUNT_TARGET, MAX_MIXED_GC_COUNT_TARGET, features); });
    threads.emplace_back([&]() { update_parameter(_old_cset_region_threshold_model, _current_old_cset_region_threshold, 
                                                  MIN_OLD_CSET_REGION_THRESHOLD, MAX_OLD_CSET_REGION_THRESHOLD, features); });
    threads.emplace_back([&]() { update_parameter(_heap_waste_percent_model, _current_heap_waste_percent, 
                                                  MIN_HEAP_WASTE_PERCENT, MAX_HEAP_WASTE_PERCENT, features); });

    for (auto& thread : threads) {
        thread.join();
    }
}
```

- 利用C++11线程库实现并行模型更新
- 每个GC参数模型在独立线程中更新，提高效率

### 2.5 GC效率反馈机制

```cpp
double G1AdaptivePolicy::calculate_gc_efficiency() {
    double reclaimed_bytes = _g1h->collection_set()->bytes_used_before() - _g1h->collection_set()->bytes_used_after();
    double gc_time_ms = _analytics->last_pause_time_ms();
    return reclaimed_bytes / gc_time_ms;
}

void G1AdaptivePolicy::update_gc_efficiency(double new_efficiency) {
    _last_gc_efficiency = new_efficiency;
    _avg_gc_efficiency = (_avg_gc_efficiency * _gc_count + new_efficiency) / (_gc_count + 1);
    double alpha = 0.2;
    _gc_efficiency_ewma = alpha * new_efficiency + (1 - alpha) * _gc_efficiency_ewma;
}
```

- 计算每次GC的效率
- 维护平均效率和指数加权移动平均效率
- 用于调整模型预测目标，提高整体GC性能

### 2.6 自适应阈值机制

```cpp
void G1AdaptivePolicy::update_parameter(AdaptiveLinearRegression& model, std::atomic<double>& current_value, 
                                        double min_value, double max_value, const std::vector<double>& features) {
    // ... 模型更新逻辑 ...
    double new_value = model.predict(selected_features);
    new_value = MAX2(min_value, MIN2(new_value, max_value));
    
    if (std::abs(new_value - current_value.load()) / current_value.load() > MODEL_UPDATE_THRESHOLD) {
        current_value.store(new_value);
    }
}
```

- 引入`MODEL_UPDATE_THRESHOLD`控制参数更新频率
- 仅当预测值与当前值的相对差异超过阈值时才更新
- 平衡了适应性和稳定性

## 3. 技术创新点

1. **在线学习模型**：采用Adam优化算法的变体，支持实时学习和适应
2. **复杂特征工程**：结合历史数据、指数移动平均和变化率，提供丰富的输入信息
3. **动态特征选择**：周期性评估和选择最相关的特征，提高模型效率
4. **并行模型更新**：利用多线程技术，提高大规模堆场景下的性能
5. **GC效率反馈**：引入GC效率指标，形成闭环优化系统
6. **自适应阈值**：通过相对变化阈值，平衡模型的响应性和稳定性

## 4. 潜在问题和解决方案

1. **模型复杂度与实时性能平衡**
   - 问题：复杂模型可能影响GC实时性能
   - 解决：优化特征选择，考虑使用增量更新算法

2. **初始阶段模型不稳定**
   - 问题：学习初期，模型可能产生不稳定预测
   - 解决：引入预训练模型或保守的初始参数设置

3. **极端场景适应性**
   - 问题：在极端GC场景下模型可能表现不佳
   - 解决：扩大训练数据范围，引入异常检测机制

4. **与现有G1框架整合**
   - 问题：新策略可能与现有G1逻辑冲突
   - 解决：设计灵活的切换机制，允许在传统和自适应策略间平滑过渡
