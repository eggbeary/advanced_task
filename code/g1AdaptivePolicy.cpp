#include "gc/g1/g1CollectedHeap.hpp"
#include "gc/g1/g1Analytics.hpp"
#include "gc/g1/g1GCPhaseTimes.hpp"
#include "runtime/os.hpp"
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>

class AdaptiveLinearRegression {
private:
    std::vector<double> _weights;
    std::vector<double> _m;
    std::vector<double> _v;
    double _beta1, _beta2, _epsilon, _alpha;
    int _t;

public:
    AdaptiveLinearRegression(size_t features, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : _weights(features, 0.0), _m(features, 0.0), _v(features, 0.0),
          _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _alpha(alpha), _t(0) {}

    double predict(const std::vector<double>& features) const {
        double result = 0.0;
        for (size_t i = 0; i < _weights.size(); ++i) {
            result += _weights[i] * features[i];
        }
        return result;
    }

    void update(const std::vector<double>& features, double target) {
        _t++;
        double prediction = predict(features);
        double error = target - prediction;
        
        for (size_t i = 0; i < _weights.size(); ++i) {
            double gradient = -error * features[i];
            _m[i] = _beta1 * _m[i] + (1 - _beta1) * gradient;
            _v[i] = _beta2 * _v[i] + (1 - _beta2) * gradient * gradient;
            double m_hat = _m[i] / (1 - std::pow(_beta1, _t));
            double v_hat = _v[i] / (1 - std::pow(_beta2, _t));
            _weights[i] -= _alpha * m_hat / (std::sqrt(v_hat) + _epsilon);
        }
    }

    const std::vector<double>& get_weights() const {
        return _weights;
    }
};

class G1AdaptivePolicy : public CHeapObj<mtGC> {
private:
    G1CollectedHeap* _g1h;
    G1Analytics*     _analytics;

    // 模型
    AdaptiveLinearRegression _mixed_gc_live_threshold_model;
    AdaptiveLinearRegression _mixed_gc_count_target_model;
    AdaptiveLinearRegression _old_cset_region_threshold_model;
    AdaptiveLinearRegression _heap_waste_percent_model;

    // 当前参数值
    std::atomic<double> _current_mixed_gc_live_threshold;
    std::atomic<uint> _current_mixed_gc_count_target;
    std::atomic<double> _current_old_cset_region_threshold;
    std::atomic<double> _current_heap_waste_percent;

    // 参数边界
    static constexpr double MIN_MIXED_GC_LIVE_THRESHOLD = 30.0;
    static constexpr double MAX_MIXED_GC_LIVE_THRESHOLD = 90.0;
    static constexpr uint MIN_MIXED_GC_COUNT_TARGET = 2;
    static constexpr uint MAX_MIXED_GC_COUNT_TARGET = 16;
    static constexpr double MIN_OLD_CSET_REGION_THRESHOLD = 5.0;
    static constexpr double MAX_OLD_CSET_REGION_THRESHOLD = 25.0;
    static constexpr double MIN_HEAP_WASTE_PERCENT = 2.0;
    static constexpr double MAX_HEAP_WASTE_PERCENT = 15.0;

    // 特征工程
    static constexpr size_t HISTORY_LENGTH = 20;
    static constexpr size_t FEATURE_COUNT = 20;
    std::deque<std::vector<double>> _feature_history;

    // 反馈循环
    double _last_gc_efficiency;
    double _avg_gc_efficiency;
    double _gc_efficiency_ewma;

    // 特征选择
    std::vector<bool> _selected_features;
    static constexpr size_t FEATURE_SELECTION_INTERVAL = 20;
    size_t _gc_count;

    // 模型更新阈值
    static constexpr double MODEL_UPDATE_THRESHOLD = 0.1;

    std::vector<double> extract_raw_features();
    std::vector<double> engineer_features(const std::vector<double>& raw_features);
    void update_feature_history(const std::vector<double>& features);
    void perform_feature_selection();

    void update_parameter(AdaptiveLinearRegression& model, std::atomic<double>& current_value, 
                          double min_value, double max_value, const std::vector<double>& features);

    double calculate_gc_efficiency();
    void update_gc_efficiency(double new_efficiency);

    // 并行处理
    void parallel_update_models(const std::vector<double>& features);

public:
    G1AdaptivePolicy(G1CollectedHeap* g1h, G1Analytics* analytics);

    void update_after_gc(const G1GCPhaseTimes& phase_times);

    double mixed_gc_live_threshold() const { return _current_mixed_gc_live_threshold.load(); }
    uint mixed_gc_count_target() const { return _current_mixed_gc_count_target.load(); }
    double old_cset_region_threshold() const { return _current_old_cset_region_threshold.load(); }
    double heap_waste_percent() const { return _current_heap_waste_percent.load(); }
};

G1AdaptivePolicy::G1AdaptivePolicy(G1CollectedHeap* g1h, G1Analytics* analytics)
    : _g1h(g1h), _analytics(analytics),
      _mixed_gc_live_threshold_model(new AdaptiveLinearRegression(FEATURE_COUNT)),
      _mixed_gc_count_target_model(new AdaptiveLinearRegression(FEATURE_COUNT)),
      _old_cset_region_threshold_model(new AdaptiveLinearRegression(FEATURE_COUNT)),
      _heap_waste_percent_model(new AdaptiveLinearRegression(FEATURE_COUNT)),
      _current_mixed_gc_live_threshold(G1MixedGCLiveThresholdPercent),
      _current_mixed_gc_count_target(G1MixedGCCountTarget),
      _current_old_cset_region_threshold(G1OldCSetRegionThresholdPercent),
      _current_heap_waste_percent(G1HeapWastePercent),
      _last_gc_efficiency(0.0),
      _avg_gc_efficiency(0.0),
      _gc_efficiency_ewma(0.0),
      _selected_features(FEATURE_COUNT, true),
      _gc_count(0) {}

std::vector<double> G1AdaptivePolicy::extract_raw_features() {
    std::vector<double> features;
    features.push_back(_g1h->used() / (double)_g1h->capacity());
    features.push_back(_analytics->predict_alloc_rate_ms());
    features.push_back(_analytics->predict_remark_time_ms());
    features.push_back(_analytics->predict_cleanup_time_ms());
    features.push_back(_g1h->collection_set()->candidates()->reclaimable_bytes() / (double)_g1h->capacity());
    features.push_back(_analytics->recent_avg_pause_time_ratio());
    features.push_back(_current_mixed_gc_live_threshold.load());
    features.push_back(_current_mixed_gc_count_target.load());
    features.push_back(_current_old_cset_region_threshold.load());
    features.push_back(_current_heap_waste_percent.load());
    features.push_back(_last_gc_efficiency);
    features.push_back(_avg_gc_efficiency);
    features.push_back(_gc_efficiency_ewma);
    // Add more features as needed
    return features;
}

std::vector<double> G1AdaptivePolicy::engineer_features(const std::vector<double>& raw_features) {
    std::vector<double> engineered_features = raw_features;

    // Add exponential moving averages
    if (!_feature_history.empty()) {
        std::vector<double> ewma(raw_features.size(), 0.0);
        double alpha = 2.0 / (_feature_history.size() + 1);
        for (size_t i = 0; i < raw_features.size(); ++i) {
            ewma[i] = raw_features[i];
            for (auto it = _feature_history.rbegin(); it != _feature_history.rend(); ++it) {
                ewma[i] = alpha * (*it)[i] + (1 - alpha) * ewma[i];
            }
            engineered_features.push_back(ewma[i]);
        }
    }

    // Add rate of change for key metrics
    if (_feature_history.size() >= 2) {
        for (size_t i = 0; i < 5; ++i) { // Consider only first 5 raw features for rate of change
            double rate = (raw_features[i] - _feature_history.front()[i]) / _feature_history.size();
            engineered_features.push_back(rate);
        }
    }

    return engineered_features;
}

void G1AdaptivePolicy::update_feature_history(const std::vector<double>& features) {
    _feature_history.push_back(features);
    if (_feature_history.size() > HISTORY_LENGTH) {
        _feature_history.pop_front();
    }
}

void G1AdaptivePolicy::perform_feature_selection() {
    std::vector<double> feature_importance(FEATURE_COUNT, 0.0);
    
    // Calculate feature importance based on the absolute values of weights across all models
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        feature_importance[i] = std::abs(_mixed_gc_live_threshold_model.get_weights()[i]) +
                                std::abs(_mixed_gc_count_target_model.get_weights()[i]) +
                                std::abs(_old_cset_region_threshold_model.get_weights()[i]) +
                                std::abs(_heap_waste_percent_model.get_weights()[i]);
    }

    // Calculate mean and standard deviation of feature importance
    double mean = 0.0, variance = 0.0;
    for (double importance : feature_importance) {
        mean += importance;
    }
    mean /= FEATURE_COUNT;

    for (double importance : feature_importance) {
        variance += (importance - mean) * (importance - mean);
    }
    variance /= FEATURE_COUNT;
    double std_dev = std::sqrt(variance);

    // Select features that are above (mean - 0.5 * std_dev)
    double threshold = mean - 0.5 * std_dev;
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        _selected_features[i] = (feature_importance[i] > threshold);
    }
}

double G1AdaptivePolicy::calculate_gc_efficiency() {
    double reclaimed_bytes = _g1h->collection_set()->bytes_used_before() - _g1h->collection_set()->bytes_used_after();
    double gc_time_ms = _analytics->last_pause_time_ms();
    return reclaimed_bytes / gc_time_ms;
}

void G1AdaptivePolicy::update_gc_efficiency(double new_efficiency) {
    _last_gc_efficiency = new_efficiency;
    
    // Update average GC efficiency
    _avg_gc_efficiency = (_avg_gc_efficiency * _gc_count + new_efficiency) / (_gc_count + 1);
    
    // Update EWMA of GC efficiency
    double alpha = 0.2; // Smoothing factor
    _gc_efficiency_ewma = alpha * new_efficiency + (1 - alpha) * _gc_efficiency_ewma;
}

void G1AdaptivePolicy::update_parameter(AdaptiveLinearRegression& model, std::atomic<double>& current_value, 
                                        double min_value, double max_value, const std::vector<double>& features) {
    std::vector<double> selected_features;
    for (size_t i = 0; i < features.size(); ++i) {
        if (_selected_features[i]) {
            selected_features.push_back(features[i]);
        }
    }

    double efficiency_factor = _gc_efficiency_ewma / _avg_gc_efficiency;
    double target = current_value.load() * std::pow(efficiency_factor, 0.5); // Adjust based on efficiency
    model.update(selected_features, target);
    double new_value = model.predict(selected_features);
    new_value = MAX2(min_value, MIN2(new_value, max_value));
    
    // Only update if the change is significant
    if (std::abs(new_value - current_value.load()) / current_value.load() > MODEL_UPDATE_THRESHOLD) {
        current_value.store(new_value);
    }
}

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

void G1AdaptivePolicy::update_after_gc(const G1GCPhaseTimes& phase_times) {
    _gc_count++;

    double new_gc_efficiency = calculate_gc_efficiency();
    update_gc_efficiency(new_gc_efficiency);

    std::vector<double> raw_features = extract_raw_features();
    std::vector<double> features = engineer_features(raw_features);
    update_feature_history(raw_features);

    parallel_update_models(features);

    if (_gc_count % FEATURE_SELECTION_INTERVAL == 0) {
        perform_feature_selection();
    }
}

G1AdaptivePolicy::~G1AdaptivePolicy() {
    delete _mixed_gc_live_threshold_model;
    delete _mixed_gc_count_target_model;
    delete _old_cset_region_threshold_model;
    delete _heap_waste_percent_model;
}

