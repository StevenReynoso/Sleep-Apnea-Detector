#ifndef APNEA_AI_H
#define APNEA_AI_H

#include <stdint.h>
#include <stdbool.h>

bool apnea_ai_init(void);
float apnea_ai_predict(const float *ecg_window, uint32_t len);

#endif
