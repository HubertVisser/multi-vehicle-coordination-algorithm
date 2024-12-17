#include <mpc_planner_modules/llm_module.h>

#include <mpc_planner_util/parameters.h>

#include <mpc_planner_parameters.h>

#include <mpc_planner_modules/llm_generated.h>

#include <ros_tools/visuals.h>

namespace MPCPlanner
{
    LLMModule::LLMModule(std::shared_ptr<Solver> solver)
        : ControllerModule(ModuleType::OBJECTIVE, solver, "LLMModule")
    {
    }

    void LLMModule::update(State &state, const RealTimeData &data, ModuleData &module_data)
    {
        (void)data;
        // Update the closest point
        double closest_s;
        _spline->findClosestPoint(state.getPos(), _closest_segment, closest_s);

        if (module_data.path.get() == nullptr && _spline.get() != nullptr)
            module_data.path = _spline;

        state.set("spline", closest_s); // We need to initialize the spline state here

        module_data.current_path_segment = _closest_segment;
    }

    void LLMModule::setParameters(const RealTimeData &data, const ModuleData &module_data, int k)
    {
        (void)data;
        (void)module_data;
        setSolverParameterGoalX(k, _solver->_params, data.goal(0));
        setSolverParameterGoalY(k, _solver->_params, data.goal(1));
        setSolverParameterReferenceVelocity(k, _solver->_params, CONFIG["weights"]["reference_velocity"].as<double>());

        // The path
        double ax, bx, cx, dx;
        double ay, by, cy, dy;
        double start;

        for (int i = 0; i < CONFIG["contouring"]["num_segments"].as<int>(); i++)
        {
            int index = _closest_segment + i;

            _spline->getParameters(index,
                                   ax, bx, cx, dx,
                                   ay, by, cy, dy);

            start = _spline->getSegmentStart(index);

            /** @note: We use the fast loading interface here as we need to load many parameters */
            setSolverParameterSplineXA(k, _solver->_params, ax, i);
            setSolverParameterSplineXB(k, _solver->_params, bx, i);
            setSolverParameterSplineXC(k, _solver->_params, cx, i);
            setSolverParameterSplineXD(k, _solver->_params, dx, i);

            setSolverParameterSplineYA(k, _solver->_params, ay, i);
            setSolverParameterSplineYB(k, _solver->_params, by, i);
            setSolverParameterSplineYC(k, _solver->_params, cy, i);
            setSolverParameterSplineYD(k, _solver->_params, dy, i);

            // Distance where this spline starts
            setSolverParameterSplineStart(k, _solver->_params, start, i);
        }

        // Obstacles
        for (size_t i = 0; i < data.dynamic_obstacles.size(); i++)
        {
            const auto &obstacle = data.dynamic_obstacles[i];
            const auto &mode = obstacle.prediction.modes[0];

            /** @note The first prediction step is index 1 of the optimization problem, i.e., k-1 maps to the predictions for this stage */
            setSolverParameterObstacleX(k, _solver->_params, mode[k - 1].position(0), i);
            setSolverParameterObstacleY(k, _solver->_params, mode[k - 1].position(1), i);
        }

        setGeneratedParameters(data, _solver, module_data, k);
    }

    void LLMModule::onDataReceived(RealTimeData &data, std::string &&data_name)
    {
        if (data_name == "reference_path")
        {
            LOG_MARK("Received Reference Path");

            // Construct a spline from the given points
            if (data.reference_path.s.empty())
                _spline = std::make_shared<RosTools::Spline2D>(data.reference_path.x, data.reference_path.y);
            else
                _spline = std::make_shared<RosTools::Spline2D>(data.reference_path.x, data.reference_path.y, data.reference_path.s);

            _closest_segment = -1;
        }
    }

    void LLMModule::visualize(const RealTimeData &data, const ModuleData &module_data)
    {
        (void)data;
        (void)module_data;

        auto &publisher = VISUALS.getPublisher(_name + "/goal");
        auto &goal = publisher.getNewPointMarker("SPHERE");
        goal.setScale(0.2, 0.2, 0.2);
        goal.setColorInt(0);
        goal.addPointMarker(Eigen::Vector2d(data.goal(0), data.goal(1)));
        publisher.publish();
    }

    // Uncomment and implement if needed

    // bool LLMModule::isDataReady(const RealTimeData &data, std::string &missing_data)
    // {
    //     // Implementation of isDataReady
    //     return true;
    // }

    // bool LLMModule::isObjectiveReached(const State &state, const RealTimeData &data)
    // {
    //     // Implementation of isObjectiveReached
    //     return true;
    // }

    // void LLMModule::reset()
    // {
    //     // Implementation of reset
    // }
}
