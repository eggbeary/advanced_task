#ifndef SHARE_GC_G1_G1ADAPTIVEPOLICY_HPP
#define SHARE_GC_G1_G1ADAPTIVEPOLICY_HPP

#include "gc/g1/g1CollectedHeap.hpp"
#include "gc/g1/g1Analytics.hpp"
#include "gc/g1/g1GCPhaseTimes.hpp"
#include "memory/allocation.hpp"
#include <vector>
#include <deque>
#include <atomic>

class AdaptiveLinearRegression;

class G1AdaptivePolicy : public CHeapObj<mtGC> {
private:
    G1CollectedHeap* _g1h;
    G1Analytics*     _analytics;

    // 模型
    AdaptiveLinearRegression* _mixed_gc_live_threshold_model;
    AdaptiveLinearRegression* _mixed_gc_count_target_model;
    AdaptiveLinearRegression* _old_cset_region_threshold_model;
    AdaptiveLinearRegression* _heap_waste_percent_model;

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

    void parallel_update_models(const std::vector<double>& features);

public:
    G1AdaptivePolicy(G1CollectedHeap* g1h, G1Analytics* analytics);
    ~G1AdaptivePolicy();

    void update_after_gc(const G1GCPhaseTimes& phase_times);

    double mixed_gc_live_threshold() const { return _current_mixed_gc_live_threshold.load(); }
    uint mixed_gc_count_target() const { return _current_mixed_gc_count_target.load(); }
    double old_cset_region_threshold() const { return _current_old_cset_region_threshold.load(); }
    double heap_waste_percent() const { return _current_heap_waste_percent.load(); }
};

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
    const std::vector<double>& get_weights() const { return _weights; }
};

#endif // SHARE_GC_G1_G1ADAPTIVEPOLICY_HPP

