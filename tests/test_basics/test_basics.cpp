#include <gtest/gtest.h>
#include <kalman_filter.h>

#include <vector>
#include <random>

TEST(KalmanFilter__init_Test, init)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.0, 0.0, 0.0},
                                                        {0.0, 0.0, 0.0},
                                                        {0.0, 0.0, 0.0}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    // Check if the state vector and the init covariance equal the initial set values.

    std::vector<double> estimate_state = kf.getEstimateState();
    std::vector<std::vector<double>> estimate_covariance = kf.getEstimateStateCovariance();

    for (unsigned int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(init_state[i], estimate_state[i]);
        for (unsigned int j = 0; j < 3; j++)
        {
            EXPECT_FLOAT_EQ(estimate_covariance[i][j], init_covariance[i][j]);
        }
    }

    // Check if the nextStatePredicted flag is set to false
    EXPECT_FALSE(kf.isNextStatePredicted());
}

TEST(KalmanFilter__init_Test, init_error)
{
    std::vector<double> init_state = {4.0, 2.0};
    std::vector<std::vector<double>> init_covariance = {{0.0, 0.0, 0.0},
                                                        {0.0, 0.0, 0.0},
                                                        {0.0, 0.0, 0.0}};

    try
    {
        KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Initial state vector and covariance matrix mismatch!");
    }
}

TEST(KalmanFilter__predict_no_control, predict)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    // Expect the next step is predicted
    EXPECT_TRUE(kf.isNextStatePredicted());

    std::vector<double> predict_state = kf.getPredictState();
    std::vector<std::vector<double>> predict_state_covariance = kf.getPredictStateCovariance();

    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(predict_state[i], init_state[i]);

        for (int j = 0; j < 3; j++)
        {
            EXPECT_FLOAT_EQ(predict_state_covariance[i][j], init_covariance[i][j]);
        }
    }
}

TEST(KalmanFilter__predict_no_control, predict_error_stm_row)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    try
    {
        kf.predict(stm, process_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: State transition matrix and state vector mismatch!");
    }
}

TEST(KalmanFilter__predict_no_control, predict_error_stm_col)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0},
                                            {0.0, 1.0},
                                            {0.0, 0.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    try
    {
        kf.predict(stm, process_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: State transition matrix and state vector mismatch!");
    }
}

TEST(KalmanFilter__predict_no_control, predict_error_pnc_row)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    try
    {
        kf.predict(stm, process_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Process noise covariance and state vector mismatch!");
    }
}

TEST(KalmanFilter__predict_no_control, predict_error_pnc_col)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0},
                                                                 {0.0, 0.0},
                                                                 {0.0, 0.0}};

    try
    {
        kf.predict(stm, process_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Process noise covariance and state vector mismatch!");
    }
}

TEST(KalmanFilter__predict_with_control, predict)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    std::vector<std::vector<double>> control_matrix = {{1.0, 0.0, 0.0},
                                                       {0.0, 1.0, 0.0},
                                                       {0.0, 0.0, 1.0}};

    std::vector<double> control_vector = {3.0, 2.0, 1.0};

    kf.predict(stm, process_noise_covariance, control_matrix, control_vector);

    // Expect the next step is predicted
    EXPECT_TRUE(kf.isNextStatePredicted());

    std::vector<double> predict_state = kf.getPredictState();
    std::vector<std::vector<double>> predict_state_covariance = kf.getPredictStateCovariance();
    // Check result

    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(predict_state[i], init_state[i] + control_vector[i]);
        for (int j = 0; j < 3; j++)
        {
            EXPECT_FLOAT_EQ(predict_state_covariance[i][j], init_covariance[i][j]);
        }
    }
}

TEST(KalmanFilter__predict_with_control, predict_error_cmat_row)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    std::vector<std::vector<double>> control_matrix = {{0.0, 1.0, 0.0},
                                                       {0.0, 0.0, 1.0}};

    std::vector<double> control_vector = {3.0, 2.0, 1.0};

    try
    {
        kf.predict(stm, process_noise_covariance, control_matrix, control_vector);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Control matrix and state vector mismatch!");
    }
}

TEST(KalmanFilter__predict_with_control, predict_error_cmat_col)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    std::vector<std::vector<double>> control_matrix = {{1.0, 0.0},
                                                       {0.0, 1.0},
                                                       {0.0, 0.0}};

    std::vector<double> control_vector = {3.0, 2.0, 1.0};

    try
    {
        kf.predict(stm, process_noise_covariance, control_matrix, control_vector);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Control input and control matrix mismatch!");
    }
}

TEST(KalmanFilter__update, update)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {{0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0}};

    kf.update(measurement_vector,
              measurement_matrix,
              measurement_noise_covariance);

    // The next state is now not predicted after the update process
    EXPECT_FALSE(kf.isNextStatePredicted());

    // Check result

    std::vector<double> estimate_state = kf.getEstimateState();
    std::vector<std::vector<double>> estimate_state_covariance = kf.getEstimateStateCovariance();

    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(estimate_state[i], init_state[i]);
        for (int j = 0; j < 3; j++)
        {
            EXPECT_FLOAT_EQ(estimate_state_covariance[i][j], 0.0);
        }
    }
}

TEST(KalmanFilter__update, update_error_not_predicted)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {{0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0}};

    try
    {
        kf.update(measurement_vector,
                  measurement_matrix,
                  measurement_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: The next state is not predicted!");
    }
}

TEST(KalmanFilter__update, update_error_mmat_row)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {{0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0}};

    try
    {
        kf.update(measurement_vector,
                  measurement_matrix,
                  measurement_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Measurement vector and measurement matrix mismatch!");
    }
}

TEST(KalmanFilter__update, update_error_mmat_col)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0},
                                                           {0.0, 1.0},
                                                           {0.0, 0.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {{0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0}};

    try
    {
        kf.update(measurement_vector,
                  measurement_matrix,
                  measurement_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Measurement matrix and state vector mismatch!");
    }
}

TEST(KalmanFilter__update, update_error_mnc_row)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {
                                                                     {0.0, 0.0, 0.0},
                                                                     {0.0, 0.0, 0.0}};

    try
    {
        kf.update(measurement_vector,
                  measurement_matrix,
                  measurement_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Measurement vector and measurement noise covariance mismatch!");
    }
}

TEST(KalmanFilter__update, update_error_mnc_col)
{
    std::vector<double> init_state = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> init_covariance = {{0.1, 0.0, 0.0},
                                                        {0.0, 0.1, 0.0},
                                                        {0.0, 0.0, 0.1}};

    KalmanFilter kf = KalmanFilter<double>(init_state, init_covariance);

    std::vector<std::vector<double>> stm = {{1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};

    std::vector<std::vector<double>> process_noise_covariance = {{0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0},
                                                                 {0.0, 0.0, 0.0}};

    kf.predict(stm, process_noise_covariance);

    std::vector<std::vector<double>> measurement_matrix = {{1.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {0.0, 0.0, 1.0}};

    std::vector<double> measurement_vector = {4.0, 2.0, 1.0};
    std::vector<std::vector<double>> measurement_noise_covariance = {{0.0, 0.0},
                                                                     {0.0, 0.0},
                                                                     {0.0, 0.0}};

    try
    {
        kf.update(measurement_vector,
                  measurement_matrix,
                  measurement_noise_covariance);
    }
    catch (const std::exception &e)
    {
        EXPECT_STREQ(e.what(), "ERROR: Measurement vector and measurement noise covariance mismatch!");
    }
}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}