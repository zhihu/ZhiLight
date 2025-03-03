#include "model/model.h"
#include <memory>
#include <stdexcept>
#include <iostream>
#include <random>

class ContextInfo {
};
typedef std::shared_ptr<ContextInfo> ContextInfoPtr;

class PyModelBase {
protected:
    std::string prefix;
    bool parallel_ { false };
    bool loaded_ { false };

    virtual void on_load() {
        loaded_ = true;
        engine()->freeze_model_memory();
    }
public:
    PyModelBase() {}
    PyModelBase(const std::string& prefix, bool parallel) : prefix(prefix), parallel_(parallel) {}

    virtual ~PyModelBase() = default;

    virtual bmengine::core::Engine* engine() = 0;

    virtual std::vector<model::ModelBase*> par_models() = 0;

    bool is_parallel() const {
        return parallel_;
    }
    bool is_loaded() const { return loaded_; }

    model::ModelBase* get_model(int i) {
        return par_models()[i];
    }

};
