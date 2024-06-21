#include <kalman_filter.h>

#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <math.h>
#include <iostream>

double measure_building(double building_height, double standard_deviation)
{
    // Generate default engine
    static std::default_random_engine rand_engine(std::time(0));

    // Generate distribution
    static std::normal_distribution<double> normal_dis(building_height, standard_deviation);

    return normal_dis(rand_engine);
}

int main()
{
    /* Set the height of the building */

    double real_building_height = 50.0;

    /* Set measurement matrix */

    std::vector<std::vector<double>> measurement_matrix = {{1.0}};

    /* Set measurement noise */

    double measurement_std_deviation = 3.0;
    std::vector<std::vector<double>> measurement_noise_covariance = {{measurement_std_deviation * measurement_std_deviation}};

    /* Give the initial guess of the building height */

    double init_guess_height = 60.0;
    double init_guess_cov = 100;

    /* Initiate kalman filter */

    KalmanFilter<double> kf({init_guess_height}, {{init_guess_cov}});

    /* Try open output file */

    std::fstream stream_output;
    try
    {
        stream_output.open("test_data", std::ios_base::out);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << '\n';
        return -1;
    }

    /* Do predict-measure-update cycle */

    unsigned int total_measure_number = 50;
    for (unsigned int i_measure = 0; i_measure < total_measure_number; i_measure++)
    {
        // State transition matrix
        std::vector<std::vector<double>> stm = {{1.0}};

        // Process noise covariance
        std::vector<std::vector<double>> pnc = {{0.0}};

        // Predict process
        kf.predict(stm, pnc);

        // Measurement process
        double measurement_height = measure_building(real_building_height, measurement_std_deviation);

        // Update process
        kf.update({measurement_height}, measurement_matrix, measurement_noise_covariance);

        /* Output result */

        stream_output << std::setw(log10(total_measure_number) + 1) << i_measure << " " 
                      << std::setw(24) << std::setprecision(16) << measurement_height << " "
                      << std::setw(24) << std::setprecision(16) << kf.getEstimateState()[0] << " "
                      << std::setw(24) << std::setprecision(16) << kf.getEstimateStateCovariance()[0][0] << std::endl;
    }

    /* Close file */

    stream_output.close();

    return 0;
}
