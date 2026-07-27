/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "string.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "engine/c_bridge.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    PredictorHandle handle;
} AdvancedPowerPredictor;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

ETH_TxPacketConfig TxConfig;
ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

ETH_HandleTypeDef heth;

UART_HandleTypeDef huart3;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ETH_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
void StartDefaultTask(void *argument);

/* USER CODE BEGIN PFP */
extern UART_HandleTypeDef huart3;

int __io_putchar(int ch) {
	HAL_UART_Transmit(&huart3, (uint8_t*) &ch, 1, HAL_MAX_DELAY);
	return ch;
}

AdvancedPowerPredictor create_advanced_power_predictor(void) {
    AdvancedPowerPredictor predictor;
    predictor.handle = create_predictor();
    // Layer 1: INT8 weights (Quantized int8 storage)
    predictor_add_layer(predictor.handle, create_linear_layer_with_dtype(5, 32, 0));
    predictor_add_layer(predictor.handle, create_batchnorm1d_layer(32, 1e-5f, 0.1f));
    predictor_add_layer(predictor.handle, create_leaky_relu_layer(0.1f));
    predictor_add_layer(predictor.handle, create_dropout_layer(0.2f));
    // Layer 2: FLOAT16 weights (Half precision IEEE 754 storage)
    predictor_add_layer(predictor.handle, create_linear_layer_with_dtype(32, 16, 3));
    predictor_add_layer(predictor.handle, create_batchnorm1d_layer(16, 1e-5f, 0.1f));
    predictor_add_layer(predictor.handle, create_leaky_relu_layer(0.1f));
    predictor_add_layer(predictor.handle, create_dropout_layer(0.2f));
    // Layer 3: FLOAT32 weights (Full precision single storage)
    predictor_add_layer(predictor.handle, create_linear_layer_with_dtype(16, 1, 4));
    return predictor;
}

void destroy_advanced_power_predictor(AdvancedPowerPredictor* predictor) {
    if (predictor && predictor->handle) {
        destroy_predictor(predictor->handle);
        predictor->handle = NULL;
    }
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define NUM_DATAPOINTS 60
#define INPUT_DIM 5
#define EPOCHS 100
#define LEARNING_RATE 0.005f
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ETH_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  /* USER CODE BEGIN 2 */

  printf("\r\n");
  printf("+------------------------------------------------------------------+\r\n");
  printf("|                STM32 EDGE TENSOR ENGINE BENCHMARK                |\r\n");
  printf("|        Architecture: ARM Cortex-M4 @ 180MHz (STM32F429ZI)        |\r\n");
  printf("|        Quantization: INT8  |  Half-FP: FLOAT16  |  FP32          |\r\n");
  printf("+------------------------------------------------------------------+\r\n\r\n");

  uint32_t cpu_time_ms = 0;
  uint32_t cmsis_time_ms = 0;
  float cpu_final_loss = 0.0f;
  float cmsis_final_loss = 0.0f;
  float cpu_prediction = 0.0f;
  float cmsis_prediction = 0.0f;

  // =========================================================================
  // PASS 1: Training WITHOUT CMSIS-NN Acceleration (Standard CPU Fallback)
  // =========================================================================
  set_cmsis_nn_acceleration_enabled(0);
  srand(42);

  float* inputs1 = (float*)malloc(NUM_DATAPOINTS * INPUT_DIM * sizeof(float));
  float* targets1 = (float*)malloc(NUM_DATAPOINTS * sizeof(float));
  if (!inputs1 || !targets1) {
      printf("CRITICAL ERROR: Memory allocation failed!\r\n");
      while(1);
  }

  for (int i = 0; i < NUM_DATAPOINTS; i++) {
      inputs1[i * INPUT_DIM + 0] = (float)(i % 365) / 365.0f;
      inputs1[i * INPUT_DIM + 1] = (float)(i % 7) / 7.0f;
      inputs1[i * INPUT_DIM + 2] = (float)(rand() % 100) / 100.0f;
      inputs1[i * INPUT_DIM + 3] = 0.5f;
      inputs1[i * INPUT_DIM + 4] = 0.5f;
      targets1[i] = (inputs1[i * INPUT_DIM + 0] * 0.4f) +
                    (inputs1[i * INPUT_DIM + 2] * 0.6f) +
                    ((float)rand() / (float)RAND_MAX * 0.05f);
  }

  AdvancedPowerPredictor predictor_cpu = create_advanced_power_predictor();

  printf("+-- [PASS 1] Standard CPU Reference Backend (No Acceleration) -----+\r\n");

  uint32_t t_start_cpu = HAL_GetTick();
  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
      cpu_final_loss = train_step(predictor_cpu.handle, inputs1, targets1, NUM_DATAPOINTS, LEARNING_RATE);
      if (epoch % 20 == 0 || epoch == EPOCHS) {
          int loss_int = (int)cpu_final_loss;
          int loss_frac = (int)((cpu_final_loss - loss_int) * 10000);
          printf("|  Epoch [%3d/%3d]  ------>  MSE Loss: %d.%04d                     |\r\n", epoch, EPOCHS, loss_int, loss_frac);
      }
  }
  cpu_time_ms = HAL_GetTick() - t_start_cpu;

  float test_input1[INPUT_DIM];
  for(int i = 0; i < INPUT_DIM; i++) test_input1[i] = inputs1[i];
  cpu_prediction = predict(predictor_cpu.handle, test_input1);

  int cpred_i = (int)cpu_prediction;
  int cpred_f = (int)((cpu_prediction - cpred_i) * 10000);
  int truth_i = (int)targets1[0];
  int truth_f = (int)((targets1[0] - truth_i) * 10000);
  printf("|  Execution Time   : %4lu ms                                      |\r\n", (unsigned long)cpu_time_ms);
  printf("|  Test Prediction  : %d.%04d (Ground Truth: %d.%04d)              |\r\n", cpred_i, cpred_f, truth_i, truth_f);
  printf("+------------------------------------------------------------------+\r\n\r\n");

  destroy_advanced_power_predictor(&predictor_cpu);
  free(inputs1);
  free(targets1);

  // =========================================================================
  // PASS 2: Training WITH CMSIS-NN Acceleration (ARM Cortex-M SIMD/DSP)
  // =========================================================================
  set_cmsis_nn_acceleration_enabled(1);
  srand(42);

  float* inputs2 = (float*)malloc(NUM_DATAPOINTS * INPUT_DIM * sizeof(float));
  float* targets2 = (float*)malloc(NUM_DATAPOINTS * sizeof(float));
  if (!inputs2 || !targets2) {
      printf("CRITICAL ERROR: Memory allocation failed!\r\n");
      while(1);
  }

  for (int i = 0; i < NUM_DATAPOINTS; i++) {
      inputs2[i * INPUT_DIM + 0] = (float)(i % 365) / 365.0f;
      inputs2[i * INPUT_DIM + 1] = (float)(i % 7) / 7.0f;
      inputs2[i * INPUT_DIM + 2] = (float)(rand() % 100) / 100.0f;
      inputs2[i * INPUT_DIM + 3] = 0.5f;
      inputs2[i * INPUT_DIM + 4] = 0.5f;
      targets2[i] = (inputs2[i * INPUT_DIM + 0] * 0.4f) +
                    (inputs2[i * INPUT_DIM + 2] * 0.6f) +
                    ((float)rand() / (float)RAND_MAX * 0.05f);
  }

  AdvancedPowerPredictor predictor_cmsis = create_advanced_power_predictor();

  printf("+-- [PASS 2] ARM CMSIS-NN Hardware Acceleration Backend (SIMD) ----+\r\n");

  uint32_t t_start_cmsis = HAL_GetTick();
  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
      cmsis_final_loss = train_step(predictor_cmsis.handle, inputs2, targets2, NUM_DATAPOINTS, LEARNING_RATE);
      if (epoch % 20 == 0 || epoch == EPOCHS) {
          int loss_int = (int)cmsis_final_loss;
          int loss_frac = (int)((cmsis_final_loss - loss_int) * 10000);
          printf("|  Epoch [%3d/%3d]  ------>  MSE Loss: %d.%04d                     |\r\n", epoch, EPOCHS, loss_int, loss_frac);
      }
  }
  cmsis_time_ms = HAL_GetTick() - t_start_cmsis;

  float test_input2[INPUT_DIM];
  for(int i = 0; i < INPUT_DIM; i++) test_input2[i] = inputs2[i];
  cmsis_prediction = predict(predictor_cmsis.handle, test_input2);

  int npred_i = (int)cmsis_prediction;
  int npred_f = (int)((cmsis_prediction - npred_i) * 10000);
  printf("|  Execution Time   : %4lu ms                                      |\r\n", (unsigned long)cmsis_time_ms);
  printf("|  Test Prediction  : %d.%04d (Ground Truth: %d.%04d)              |\r\n", npred_i, npred_f, truth_i, truth_f);
  printf("+------------------------------------------------------------------+\r\n\r\n");

  destroy_advanced_power_predictor(&predictor_cmsis);
  free(inputs2);
  free(targets2);

  // =========================================================================
  // BENCHMARK COMPARISON REPORT
  // =========================================================================
  printf("+------------------------------------------------------------------+\r\n");
  printf("|                   BENCHMARK EXECUTION SUMMARY                    |\r\n");
  printf("+----------------------+-+--------------+-+--------------+-+-------+\r\n");
  printf("| Backend Mode         | | Exec Time    | | Final MSE    | | Output|\r\n");
  printf("+----------------------+-+--------------+-+--------------+-+-------+\r\n");

  int closs_i = (int)cpu_final_loss;
  int closs_f = (int)((cpu_final_loss - closs_i) * 10000);
  printf("| CPU Reference        | | %4lu ms       | | %d.%04d       | | %d.%04d|\r\n", (unsigned long)cpu_time_ms, closs_i, closs_f, cpred_i, cpred_f);

  int nloss_i = (int)cmsis_final_loss;
  int nloss_f = (int)((cmsis_final_loss - nloss_i) * 10000);
  printf("| CMSIS-NN Accelerated | | %4lu ms       | | %d.%04d       | | %d.%04d|\r\n", (unsigned long)cmsis_time_ms, nloss_i, nloss_f, npred_i, npred_f);

  printf("+----------------------+-+--------------+-+--------------+-+-------+\r\n\r\n");

  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ETH Initialization Function
  * @param None
  * @retval None
  */
static void MX_ETH_Init(void)
{

  /* USER CODE BEGIN ETH_Init 0 */

  /* USER CODE END ETH_Init 0 */

   static uint8_t MACAddr[6];

  /* USER CODE BEGIN ETH_Init 1 */

  /* USER CODE END ETH_Init 1 */
  heth.Instance = ETH;
  MACAddr[0] = 0x00;
  MACAddr[1] = 0x80;
  MACAddr[2] = 0xE1;
  MACAddr[3] = 0x00;
  MACAddr[4] = 0x00;
  MACAddr[5] = 0x00;
  heth.Init.MACAddr = &MACAddr[0];
  heth.Init.MediaInterface = HAL_ETH_RMII_MODE;
  heth.Init.TxDesc = DMATxDscrTab;
  heth.Init.RxDesc = DMARxDscrTab;
  heth.Init.RxBuffLen = 1524;

  /* USER CODE BEGIN MACADDRESS */

  /* USER CODE END MACADDRESS */

  if (HAL_ETH_Init(&heth) != HAL_OK)
  {
    Error_Handler();
  }

  memset(&TxConfig, 0 , sizeof(ETH_TxPacketConfig));
  TxConfig.Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD;
  TxConfig.ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC;
  TxConfig.CRCPadCtrl = ETH_CRC_PAD_INSERT;
  /* USER CODE BEGIN ETH_Init 2 */

  /* USER CODE END ETH_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_FS_PCD_Init(void)
{

  /* USER CODE BEGIN USB_OTG_FS_Init 0 */

  /* USER CODE END USB_OTG_FS_Init 0 */

  /* USER CODE BEGIN USB_OTG_FS_Init 1 */

  /* USER CODE END USB_OTG_FS_Init 1 */
  hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
  hpcd_USB_OTG_FS.Init.dev_endpoints = 4;
  hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_OTG_FS_Init 2 */

  /* USER CODE END USB_OTG_FS_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN 5 */
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END 5 */
}

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM7 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM7)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
