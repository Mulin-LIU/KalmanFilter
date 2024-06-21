#include <kalman_filter.h>

#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <math.h>
#include <iostream>

#define MOVING_RADIUS 10.0
#define MOVING_PERIOD 10.0
#define MOVING_ANGULAR_VELOCITY (2 * M_PI / MOVING_PERIOD)

std::vector<double> vehicle_real_state(double t)
{
    std::vector<double> result(6);

    double theta = MOVING_ANGULAR_VELOCITY * t;
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    double x = MOVING_RADIUS * cos_theta;
    double y = MOVING_RADIUS * sin_theta;
    double vx = -MOVING_ANGULAR_VELOCITY * MOVING_RADIUS * sin_theta;
    double vy = MOVING_ANGULAR_VELOCITY * MOVING_RADIUS * cos_theta;
    double ax = -MOVING_ANGULAR_VELOCITY * MOVING_ANGULAR_VELOCITY * x;
    double ay = -MOVING_ANGULAR_VELOCITY * MOVING_ANGULAR_VELOCITY * y;

    result[0] = x;
    result[1] = y;
    result[2] = vx;
    result[3] = vy;
    result[4] = ax;
    result[5] = ay;

    return result;
}

std::vector<double> measure_real_state(std::vector<double> state,
                                       double measurement_std_deviation_x,
                                       double measurement_std_deviation_y)
{
    // Generate default engine
    static std::default_random_engine rand_engine(std::time(0));

    // Generate distribution
    static std::normal_distribution<double> normal_dis_x(0.0, measurement_std_deviation_x);
    static std::normal_distribution<double> normal_dis_y(0.0, measurement_std_deviation_y);

    std::vector<double> result(2);
    result[0] = state[0] + normal_dis_x(rand_engine);
    result[1] = state[1] + normal_dis_y(rand_engine);

    return result;
}

int main()
{
    /* Set the vehicle's initial state */

    // The initial guess of state (x, y, vx, vy, ax, ay)
    std::vector<double> init_state_guess = vehicle_real_state(0);
    init_state_guess[0] = init_state_guess[0] + 1.2;
    init_state_guess[1] = init_state_guess[1] + 0.5;
    init_state_guess[2] = init_state_guess[2] + 0.8;
    init_state_guess[3] = init_state_guess[3] + 0.2;
    init_state_guess[4] = 0.0;
    init_state_guess[5] = 0.0;

    // The initial state covariance
    std::vector<std::vector<double>> init_state_guess_covariance = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                    {0.0, 4.0, 0.0, 0.0, 0.0, 0.0},
                                                                    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                                                    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                                                    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                                                    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

    /* Set measurement setups */

    // The measurement matrix
    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}};

    double measurement_std_dev_x = 1.0;
    double measurement_std_dev_y = 1.0;

    // The measurement noise covariance
    std::vector<std::vector<double>> measurement_noise_covariance = {{measurement_std_dev_x * measurement_std_dev_x, 0.0},
                                                                     {0.0, measurement_std_dev_y * measurement_std_dev_y}};

    // The measurement interval
    double measurement_dt = MOVING_PERIOD / 100;
    double mdt = measurement_dt;
    double dt2 = mdt * mdt;

    /* Set other parameters */

    std::vector<std::vector<double>> state_transition_matrix = {{1.0, 0.0, mdt, 0.0, 0.5 * dt2, 0.0},
                                                                {0.0, 1.0, 0.0, mdt, 0.0, 0.5 * dt2},
                                                                {0.0, 0.0, 1.0, 0.0, mdt, 0.0},
                                                                {0.0, 0.0, 0.0, 1.0, 0.0, mdt},
                                                                {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                                                {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    /* Initiate kalman filter */

    KalmanFilter<double> kf(init_state_guess, init_state_guess_covariance);

    /* Try open output file path */

    std::fstream stream_output;
    try
    {
        stream_output.open("./test_data", std::ios_base::out);
    }
    catch (const std::exception &e)
    {
        return -1;
    }

    /* Begin iterations */

    unsigned int total_measurement = 100;

    for (unsigned int i_measure = 1; i_measure < total_measurement; i_measure++)
    {
        // Current time
        double current_t = i_measure * measurement_dt;

        // Get real state
        std::vector<double> real_state = vehicle_real_state(current_t);

        // Get state measurement
        std::vector<double> measure_state = measure_real_state(real_state, measurement_std_dev_x, measurement_std_dev_y);

        // Predict
        kf.predict(state_transition_matrix, process_noise_covariance);

        // Update
        kf.update(measure_state, measurement_matrix, measurement_noise_covariance);

        // Get predict and estimate state

        std::vector<double> predict_state = kf.getPredictState();
        std::vector<double> estimate_state = kf.getEstimateState();
        std::vector<std::vector<double>> estimate_state_covariance = kf.getEstimateStateCovariance();

        // Output result
        stream_output << std::setw(log10(total_measurement) + 1) << i_measure << " ";
        for (int i_state = 0; i_state < real_state.size(); i_state++)
        {
            stream_output << std::setw(24) << std::setprecision(16) << real_state[i_state] << " ";
        }
        stream_output << std::endl;
        stream_output << std::setw(log10(total_measurement) + 1) << "" << " ";
        for (int i_state = 0; i_state < measure_state.size(); i_state++)
        {
            stream_output << std::setw(24) << std::setprecision(16) << measure_state[i_state] << " ";
        }
        stream_output << std::endl;
        stream_output << std::setw(log10(total_measurement) + 1) << "" << " ";
        for (int i_state = 0; i_state < estimate_state.size(); i_state++)
        {
            stream_output << std::setw(24) << std::setprecision(16) << estimate_state[i_state] << " ";
        }
        stream_output << std::endl;
        stream_output << std::setw(log10(total_measurement) + 1) << "" << " ";
        for (int i_state = 0; i_state < estimate_state.size(); i_state++)
        {
            stream_output << std::setw(24) << std::setprecision(16) << estimate_state_covariance[i_state][i_state] << " ";
        }
        stream_output << std::endl;
    }

    /* Close output stream */

    stream_output.close();

    return 0;
}