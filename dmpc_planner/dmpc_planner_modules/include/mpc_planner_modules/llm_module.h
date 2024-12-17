#ifndef __LLMModule_H_
#define __LLMModule_H_

#include <mpc_planner_modules/controller_module.h>

#include <ros_tools/spline.h>

namespace MPCPlanner
{
  class LLMModule : public ControllerModule
  {
  public:
    LLMModule(std::shared_ptr<Solver> solver);

  public:
    void update(State &state, const RealTimeData &data, ModuleData &module_data) override;
    void setParameters(const RealTimeData &data, const ModuleData &module_data, int k) override;

    void onDataReceived(RealTimeData &data, std::string &&data_name) override;
    // bool isDataReady(const RealTimeData &data, std::string &missing_data) override;

    // bool isObjectiveReached(const State &state, const RealTimeData &data) override;

    void visualize(const RealTimeData &data, const ModuleData &module_data) override;

    // void reset() override;

  private:
    int _closest_segment{-1};
    std::shared_ptr<RosTools::Spline2D> _spline{nullptr};

  };
}
#endif // __LLMModule_H_