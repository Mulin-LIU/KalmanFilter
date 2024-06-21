#pragma once

#include <vector>

template <typename T>
class KalmanFilter
{
private:
    /// @brief Supplementary functions
    class SupplementFunctions
    {
    private:
    public:
        static std::vector<T> vector_minus(const std::vector<T> &vector1, const std::vector<T> &vector2)
        {
            std::vector<T> result(vector1.size());
            for (int i = 0; i < vector1.size(); i++)
            {
                result[i] = vector1[i] - vector2[i];
            }

            return result;
        }

        static std::vector<T> vector_plus(const std::vector<T> &vector1, const std::vector<T> &vector2)
        {
            std::vector<T> result(vector1.size());
            for (int i = 0; i < vector1.size(); i++)
            {
                result[i] = vector1[i] + vector2[i];
            }

            return result;
        }

        static std::vector<T> matrix_multiply_vector(const std::vector<std::vector<T>> &matrix, const std::vector<T> &vector)
        {
            unsigned int m_matrix = matrix.size();
            unsigned int n_matrix = matrix[0].size();
            unsigned int n_vector = vector.size();

            std::vector<T> result(m_matrix);
            for (unsigned int i = 0; i < m_matrix; i++)
            {
                result[i] = 0;
                for (unsigned int j = 0; j < n_matrix; j++)
                {
                    result[i] = result[i] + matrix[i][j] * vector[j];
                }
            }

            return result;
        }

        static std::vector<std::vector<T>> matrix_multiply_matrix(const std::vector<std::vector<T>> &matrix1, const std::vector<std::vector<T>> &matrix2)
        {
            unsigned int m_matrix1 = matrix1.size();
            unsigned int n_matrix1 = matrix1[0].size();
            unsigned int n_matrix2 = matrix2.size();
            unsigned int p_matrix2 = matrix2[0].size();

            std::vector<std::vector<T>> result(m_matrix1);
            for (unsigned int i = 0; i < m_matrix1; i++)
            {
                result[i].resize(p_matrix2);
                for (unsigned int j = 0; j < p_matrix2; j++)
                {
                    result[i][j] = 0;
                    for (unsigned int k = 0; k < n_matrix1; k++)
                    {
                        result[i][j] = result[i][j] + matrix1[i][k] * matrix2[k][j];
                    }
                }
            }

            return result;
        }

        static std::vector<std::vector<T>> matrix_transpose(const std::vector<std::vector<T>> &matrix)
        {
            unsigned int m_matrix = matrix.size();
            unsigned int n_matrix = matrix[0].size();

            std::vector<std::vector<T>> result(n_matrix);
            for (unsigned int i = 0; i < n_matrix; i++)
            {
                result[i].resize(m_matrix);
                for (unsigned int j = 0; j < m_matrix; j++)
                {
                    result[i][j] = matrix[j][i];
                }
            }

            return result;
        }

        static std::vector<std::vector<T>> matrix_plus_matrix(const std::vector<std::vector<T>> &matrix1, const std::vector<std::vector<T>> &matrix2)
        {
            unsigned int m_matrix1 = matrix1.size();
            unsigned int n_matrix1 = matrix1[0].size();

            std::vector<std::vector<T>> result(m_matrix1);
            for (unsigned int iRow = 0; iRow < m_matrix1; iRow++)
            {
                result[iRow].resize(n_matrix1);
                for (unsigned int iCol = 0; iCol < n_matrix1; iCol++)
                {
                    result[iRow][iCol] = matrix1[iRow][iCol] + matrix2[iRow][iCol];
                }
            }

            return result;
        }

        static std::vector<std::vector<T>> matrix_inverse(std::vector<std::vector<T>> matrix, unsigned int order)
        {
            std::vector<std::vector<T>> result(order);

            for (unsigned int iRow = 0; iRow < order; iRow++)
            {
                result[iRow].resize(order);
            }

            // make the result matrix identity matrix
            for (unsigned int iRow = 0; iRow < order; iRow++)
            {
                for (unsigned int iCol = 0; iCol < order; iCol++)
                {
                    if (iRow == iCol)
                    {
                        result[iRow][iCol] = 1;
                    }
                    else
                    {
                        result[iRow][iCol] = 0;
                    }
                }
            }

            // Collumn traversal
            for (unsigned int j = 0; j < order; j++)
            {
                // Find the maximum in this collumn
                T tempMax = 0;
                unsigned int tempIdx = 0;
                for (unsigned int i = j; i < order; i++)
                {
                    if (fabs(matrix[i][j]) > tempMax)
                    {
                        tempIdx = i;
                        tempMax = fabs(matrix[i][j]);
                    }
                    else
                    {
                        ;
                    }
                }

                // Swap rows
                if (j == tempIdx)
                {
                    ;
                }
                else
                {
                    for (unsigned int jj = 0; jj < order; jj++)
                    {
                        // Swap matrix
                        T temp = matrix[j][jj];
                        matrix[j][jj] = matrix[tempIdx][jj];
                        matrix[tempIdx][jj] = temp;

                        // Swap inverse
                        temp = result[j][jj];
                        result[j][jj] = result[tempIdx][jj];
                        result[tempIdx][jj] = temp;
                    }
                }

                // eliminate elements not on the trace
                for (unsigned int i = 0; i < order; i++)
                {
                    if (i == j)
                    {
                        ;
                    }
                    else
                    {
                        T tempFactor = -matrix[i][j] / matrix[j][j];
                        for (unsigned int jj = 0; jj < order; jj++)
                        {
                            matrix[i][jj] = matrix[i][jj] + tempFactor * matrix[j][jj];
                            result[i][jj] = result[i][jj] + tempFactor * result[j][jj];
                        }
                    }
                }
            }

            // make the trace element 1
            for (unsigned int i = 0; i < order; i++)
            {
                for (unsigned int j = 0; j < order; j++)
                {
                    result[i][j] = result[i][j] / matrix[i][i];
                }
            }

            return result;
        }
    };

    /// @brief Estimated state
    std::vector<T> mEstimateState;

    /// @brief Covariance of the estimated state
    std::vector<std::vector<T>> mEstimateStateCovariance;

    /// @brief Pridicted state
    std::vector<T> mPredictState;

    /// @brief Covariance of the predicted state
    std::vector<std::vector<T>> mPredictStateCovariance;

    /// @brief If the next state is predicted
    bool mNextStatePredicted;

public:
    /// @brief The constructor without argument is deleted
    KalmanFilter() = delete;

    /// @brief The constructor that initiates the Kalman filter
    /// @param initialState The initial state vector
    /// @param initialStateCovariance The initial state covariance
    KalmanFilter(const std::vector<T> &initialState,
                 const std::vector<std::vector<T>> &initialStateCovariance)
    {
        setInitialState(initialState, initialStateCovariance);
    }

    /// @brief To calculate the next predicted state vector and covariance matrix
    /// @param stateTransitionMatrix The state transition matrix
    /// @param processNoiseCovariance The process noise covariance matrix
    void predict(const std::vector<std::vector<T>> &stateTransitionMatrix,
                 const std::vector<std::vector<T>> &processNoiseCovariance)
    {
        // Check state transition matrix

        unsigned int stmDimensionRow = stateTransitionMatrix.size();
        unsigned int stateDimension = mEstimateState.size();
        if (stmDimensionRow != stateDimension)
        {
            throw std::invalid_argument("ERROR: State transition matrix and state vector mismatch!");
        }

        for (unsigned int istmRow = 0; istmRow < stmDimensionRow; istmRow++)
        {
            if (stateTransitionMatrix[istmRow].size() != stateDimension)
            {
                throw std::invalid_argument("ERROR: State transition matrix and state vector mismatch!");
            }
        }

        // Check process noise covariance

        unsigned int processNoiseCovarianceDimensionRow = processNoiseCovariance.size();
        if (processNoiseCovarianceDimensionRow != stateDimension)
        {
            throw std::invalid_argument("ERROR: Process noise covariance and state vector mismatch!");
        }

        for (unsigned int iProcessNoiseCovRow = 0; iProcessNoiseCovRow < processNoiseCovarianceDimensionRow; iProcessNoiseCovRow++)
        {
            if (processNoiseCovariance[iProcessNoiseCovRow].size() != stateDimension)
            {
                throw std::invalid_argument("ERROR: Process noise covariance and state vector mismatch!");
            }
        }

        // Predict the next state
        // x' = Fx

        mPredictState = SupplementFunctions::matrix_multiply_vector(stateTransitionMatrix, mEstimateState);

        // Predict the next state covariance
        // P' = FPF^T + Q

        mPredictStateCovariance = SupplementFunctions::matrix_multiply_matrix(stateTransitionMatrix, mEstimateStateCovariance);
        mPredictStateCovariance = SupplementFunctions::matrix_multiply_matrix(mPredictStateCovariance,
                                                                              SupplementFunctions::matrix_transpose(stateTransitionMatrix));
        mPredictStateCovariance = SupplementFunctions::matrix_plus_matrix(mPredictStateCovariance, processNoiseCovariance);

        // Set mNextStatePredicted to true
        mNextStatePredicted = true;
    }

    /// @brief To calculate the next predicted state vector and covariance matrix. Input control is considered.
    /// @param stateTransitionMatrix The state transition matrix
    /// @param processNoiseCovariance The process noise covariance matrix
    /// @param controlMatrix The control matrix
    /// @param controlInput The control input vector
    void predict(const std::vector<std::vector<T>> &stateTransitionMatrix,
                 const std::vector<std::vector<T>> &processNoiseCovariance,
                 const std::vector<std::vector<T>> &controlMatrix,
                 const std::vector<T> &controlInput)
    {
        // Check control input

        unsigned int stateDimension = mEstimateState.size();

        unsigned int controlMatrixDimensionRow = controlMatrix.size();
        unsigned int controlInputDimenstion = controlInput.size();

        if (controlMatrixDimensionRow != stateDimension)
        {
            throw std::invalid_argument("ERROR: Control matrix and state vector mismatch!");
        }

        for (unsigned int iControlMatrixRow = 0; iControlMatrixRow < controlMatrixDimensionRow; iControlMatrixRow++)
        {
            if (controlMatrix[iControlMatrixRow].size() != controlInputDimenstion)
            {
                throw std::invalid_argument("ERROR: Control input and control matrix mismatch!");
            }
        }

        // Call predict without control

        predict(stateTransitionMatrix, processNoiseCovariance);

        // Calculate control vector
        // control = Gu

        std::vector<T> controlVector = SupplementFunctions::matrix_multiply_vector(controlMatrix, controlInput);

        // x' = Fx + Gu

        mPredictState = SupplementFunctions::vector_plus(controlVector, mPredictState);

        // Set mNextStatePredicted to true

        mNextStatePredicted = true;
    }

    /// @brief To update the estimated state and estimated state covariance based on former predicted
    ///        state and state covariance. The predict step should be called before this function is
    ///        implemented.
    /// @param measurementVector The measurement vector
    /// @param measurementMatrix The measurement matrix
    /// @param measurementNoiseCovariance The measurement noise covariance
    void update(const std::vector<T> &measurementVector,
                const std::vector<std::vector<T>> &measurementMatrix,
                const std::vector<std::vector<T>> &measurementNoiseCovariance)
    {
        // Check if state predicted forehead

        if (!mNextStatePredicted)
        {
            throw std::runtime_error("ERROR: The next state is not predicted!");
        }

        // Check measurement inputs

        unsigned int measurementVectorDimension = measurementVector.size();
        unsigned int measurementMatrixDimensionRow = measurementMatrix.size();
        unsigned int measurementNoiseCovDimensionRow = measurementNoiseCovariance.size();

        if (measurementVectorDimension != measurementMatrixDimensionRow)
        {
            throw std::invalid_argument("ERROR: Measurement vector and measurement matrix mismatch!");
        }

        for (unsigned int iMeasurementMatRow = 0; iMeasurementMatRow < measurementMatrixDimensionRow; iMeasurementMatRow++)
        {
            if (measurementMatrix[iMeasurementMatRow].size() != mEstimateState.size())
            {
                throw std::invalid_argument("ERROR: Measurement matrix and state vector mismatch!");
            }
        }

        if (measurementNoiseCovDimensionRow != measurementVectorDimension)
        {
            throw std::invalid_argument("ERROR: Measurement vector and measurement noise covariance mismatch!");
        }

        for (unsigned int iMeasurementNoiseCovRow = 0; iMeasurementNoiseCovRow < measurementNoiseCovDimensionRow; iMeasurementNoiseCovRow++)
        {
            if (measurementNoiseCovariance[iMeasurementNoiseCovRow].size() != measurementVectorDimension)
            {
                throw std::invalid_argument("ERROR: Measurement vector and measurement noise covariance mismatch!");
            }
        }

        // Calculate Karman gain
        // K = P' H^T (H P' H^T + R)^-1

        std::vector<std::vector<T>> tempMat = SupplementFunctions::matrix_transpose(measurementMatrix); // H^T
        std::vector<std::vector<T>> P_HT = SupplementFunctions::matrix_multiply_matrix(mPredictStateCovariance, tempMat);

        tempMat = SupplementFunctions::matrix_multiply_matrix(measurementMatrix, P_HT);         // H P H^T
        tempMat = SupplementFunctions::matrix_plus_matrix(tempMat, measurementNoiseCovariance); // H P H^T + R
        tempMat = SupplementFunctions::matrix_inverse(tempMat, tempMat.size());                 // (H P H^T + R)^-1

        std::vector<std::vector<T>> kalmanGain = SupplementFunctions::matrix_multiply_matrix(P_HT, tempMat);

        // Update state vector
        // x' = x + K(z - Hx)

        std::vector<T> tempVec = SupplementFunctions::matrix_multiply_vector(measurementMatrix, mPredictState); // Hx
        tempVec = SupplementFunctions::vector_minus(measurementVector, tempVec);                                // z - Hx
        tempVec = SupplementFunctions::matrix_multiply_vector(kalmanGain, tempVec);                             // K(z - Hx)

        mEstimateState = SupplementFunctions::vector_plus(mPredictState, tempVec);

        // Update state covariance
        // P' = (I - KH) P (I - KH)^T + K R K^T

        tempMat = SupplementFunctions::matrix_multiply_matrix(kalmanGain, measurementMatrix); // KH
        for (unsigned int i = 0; i < tempMat.size(); i++)                                     // I - KH
        {
            for (unsigned int j = 0; j < tempMat.size(); j++)
            {
                if (i == j)
                {
                    tempMat[i][j] = 1 - tempMat[i][j];
                }
                else
                {
                    tempMat[i][j] = -tempMat[i][j];
                }
            }
        }

        std::vector<std::vector<T>> tempMat2 = SupplementFunctions::matrix_transpose(tempMat);   // (I - KH)^T
        tempMat = SupplementFunctions::matrix_multiply_matrix(tempMat, mPredictStateCovariance); // (I - KH) P
        tempMat = SupplementFunctions::matrix_multiply_matrix(tempMat, tempMat2);                // (I - KH) P (I - KH)^T

        tempMat2 = SupplementFunctions::matrix_multiply_matrix(kalmanGain, measurementNoiseCovariance); // K R
        tempMat2 = SupplementFunctions::matrix_multiply_matrix(tempMat2,
                                                               SupplementFunctions::matrix_transpose(kalmanGain)); // K R K^T

        mEstimateStateCovariance = SupplementFunctions::matrix_plus_matrix(tempMat, tempMat2);

        // Set next state predicted to false
        mNextStatePredicted = false;
    }

    /// @brief To get the estimate state
    /// @return
    std::vector<T> getEstimateState()
    {
        return mEstimateState;
    }

    /// @brief To get the next predicted state
    /// @return
    std::vector<T> getPredictState()
    {
        return mPredictState;
    }

    /// @brief To get the estimate state covariance
    /// @return
    std::vector<std::vector<T>> getEstimateStateCovariance()
    {
        return mEstimateStateCovariance;
    }

    /// @brief To get the predict state covariance
    /// @return
    std::vector<std::vector<T>> getPredictStateCovariance()
    {
        return mPredictStateCovariance;
    }

    /// @brief To check if the next state is predicted
    /// @return
    bool isNextStatePredicted()
    {
        return mNextStatePredicted;
    }

    /// @brief To set the initial state and the initial state covariance
    /// @param initialState
    /// @param initialStateCovariance
    void setInitialState(const std::vector<T> &initialState,
                         const std::vector<std::vector<T>> &initialStateCovariance)
    {
        // Check input parameters

        unsigned int stateDimension = initialState.size();
        unsigned int stateCovarianceDimensionRow = initialStateCovariance.size();

        if (stateDimension != stateCovarianceDimensionRow)
        {
            throw std::invalid_argument("ERROR: Initial state vector and covariance matrix mismatch!");
        }

        for (unsigned int iCovarianceRow = 0; iCovarianceRow < stateCovarianceDimensionRow; iCovarianceRow++)
        {
            if (initialStateCovariance[iCovarianceRow].size() != stateDimension)
            {
                throw std::invalid_argument("ERROR: Initial state vector and covariance matrix mismatch!");
            }
        }

        // Set values

        mEstimateState = initialState;
        mEstimateStateCovariance = initialStateCovariance;
        mNextStatePredicted = false;
    }
};