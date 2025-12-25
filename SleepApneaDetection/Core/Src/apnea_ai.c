#include "apnea_ai.h"
#include "network.h"
#include "network_data.h"

/* Static network handle */
static ai_handle s_network = AI_HANDLE_NULL;

/* Activations buffer – size from network_data.h */
AI_ALIGNED(4)
static ai_u8 s_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* Output buffer – size from network.h (should be 1 for binary output) */
AI_ALIGNED(4)
static float s_output_data[AI_NETWORK_OUT_1_SIZE];

bool apnea_ai_init(void)
{
    ai_error err;

    /* Create network instance */
    err = ai_network_create(&s_network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        return false;
    }

    /* Initialize network params: weights + activations */
    const ai_network_params params = {
        .params      = AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        .activations = AI_NETWORK_DATA_ACTIVATIONS(s_activations),
    };

    if (!ai_network_init(s_network, &params)) {
        err = ai_network_get_error(s_network);
        (void)err;  /* you can printf/log later if you care */
        return false;
    }

    return true;
}

float apnea_ai_predict(const float *ecg_window, uint32_t len)
{
    if (s_network == AI_HANDLE_NULL) {
        return -1.0f;  // not initialized
    }

    if (!ecg_window || len != AI_NETWORK_IN_1_SIZE) {
        return -1.0f;  // wrong input size
    }

    ai_u16 n_in = 0;
    ai_u16 n_out = 0;

    /* Get runtime input/output buffer descriptors from the network */
    ai_buffer *ai_input  = ai_network_inputs_get(s_network, &n_in);
    ai_buffer *ai_output = ai_network_outputs_get(s_network, &n_out);

    if (!ai_input || !ai_output || n_in < 1 || n_out < 1) {
        return -1.0f;
    }

    /* Point input buffer to your window */
    ai_input[0].data = AI_HANDLE_PTR((void *)ecg_window);

    /* Point output buffer to our static float array */
    ai_output[0].data = AI_HANDLE_PTR(s_output_data);

    /* Run the network */
    ai_i32 nbatch = ai_network_run(s_network, ai_input, ai_output);
    if (nbatch != 1) {
        ai_error err = ai_network_get_error(s_network);
        (void)err;
        return -1.0f;
    }

    /* Single-neuron sigmoid output */
    return s_output_data[0];
}
