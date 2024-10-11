/**
* @file model_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "model_process.h"
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include "utils.h"

using namespace std;
extern bool g_isDevice;

ModelProcess::ModelProcess() :modelId_(0), modelWorkSize_(0), modelWeightSize_(0), modelWorkPtr_(nullptr),
    modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr)
{
}

ModelProcess::~ModelProcess()
{
    UnloadModel();
    DestroyModelDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModel(const char *modelPath)
{
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }
    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
            modelWorkSize_, static_cast<int32_t>(ret));
        return FAILED;
    }

    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
            modelWeightSize_, static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
        modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::CreateModelDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyModelDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("destroy model description success");
}

Result ModelProcess::GetInputSizeByIndex(const size_t index, size_t &inputSize)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    inputSize = aclmdlGetInputSizeByIndex(modelDesc_, index);
    return SUCCESS;
}

Result ModelProcess::CreateInput()
{
    // om used in this sample has only one input
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }

    size_t templateSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    void *templateHost;
    aclError ret = aclrtMallocHost(&templateHost, templateSize);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("can't malloc host template, size is %zu, create template failed, errorCode is %d",
            templateSize, static_cast<int32_t>(ret));
        return FAILED;
    }
    for (int i = 0; i < templateSize / sizeof(float); i++) {
        reinterpret_cast<float*>(templateHost)[i] = 3;
    }

    void *templateBuffer = nullptr;
    ret = aclrtMalloc(&templateBuffer, templateSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("can't malloc buffer, size is %zu, create template failed, errorCode is %d",
            templateSize, static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclrtMemcpy(templateBuffer, templateSize, templateHost, templateSize,
                      ACL_MEMCPY_HOST_TO_DEVICE
                          );
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy template from host to device error, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    size_t searchSize = aclmdlGetInputSizeByIndex(modelDesc_, 1);
    void *searchHost;
    ret = aclrtMallocHost(&searchHost, searchSize);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("can't malloc host search, size is %zu, create template failed, errorCode is %d",
            searchSize, static_cast<int32_t>(ret));
        return FAILED;
    }

    for (int i = 0; i < searchSize / sizeof(float); i++) {
        reinterpret_cast<float*>(templateHost)[i] = 3;
    }

    void *searchBuffer = nullptr;
    ret = aclrtMalloc(&searchBuffer, searchSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("can't malloc buffer, size is %zu, create search failed, errorCode is %d",
            searchSize, static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclrtMemcpy(searchBuffer, searchSize, searchHost, searchSize,
                      ACL_MEMCPY_HOST_TO_DEVICE
                          );
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy search from host to device error, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    aclDataBuffer *templateData = aclCreateDataBuffer(templateBuffer, templateSize);
    if (templateData == nullptr) {
        ERROR_LOG("can't create data buffer, create template failed");
        return FAILED;
    }

    aclDataBuffer *search = aclCreateDataBuffer(searchBuffer, searchSize);
    if (search == nullptr) {
        ERROR_LOG("can't create data buffer, create search failed");
        return FAILED;
    }

    ret = aclmdlAddDatasetBuffer(input_, templateData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(templateData);
        templateData = nullptr;
        return FAILED;
    }

    ret = aclmdlAddDatasetBuffer(input_, search);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(search);
        search = nullptr;
        return FAILED;
    }
    INFO_LOG("create model input success");

    // INFO_LOG("input num: %zu, template size: %zu, seach size: %zu", aclmdlGetNumInputs(modelDesc_), templateSize, searchSize);

    return SUCCESS;
}

void ModelProcess::DestroyInput()
{
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success");
}

Result ModelProcess::CreateOutput()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    // INFO_LOG("output num: %zu", outputSize);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        // INFO_LOG("  %zu", modelOutputSize);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
                static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");

    return SUCCESS;
}

void ModelProcess::DumpModelOutputResult()
{
    stringstream ss;
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);
    static int executeNum = 0;
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << ++executeNum << "_" << i << ".bin";
        string outputFileName = ss.str();
        FILE *outputFile = fopen(outputFileName.c_str(), "wb");
        if (outputFile != nullptr) {
            // get model output data
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void *data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

            void *outHostData = nullptr;
            aclError ret = ACL_SUCCESS;
            if (!g_isDevice) {
                ret = aclrtMallocHost(&outHostData, len);
                if (ret != ACL_SUCCESS) {
                    ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                        len, static_cast<int32_t>(ret));
                    fclose(outputFile);
                    return;
                }

                // if app is running in host, need copy model output data from device to host
                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                    (void)aclrtFreeHost(outHostData);
                    fclose(outputFile);
                    return;
                }

                fwrite(outHostData, len, sizeof(char), outputFile);

                ret = aclrtFreeHost(outHostData);
                if (ret != ACL_SUCCESS) {
                    ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                    fclose(outputFile);
                    return;
                }
            } else {
                // if app is running in host, write model output data into result file
                fwrite(data, len, sizeof(char), outputFile);
            }
            fclose(outputFile);
        } else {
            ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
            return;
        }
    }

    INFO_LOG("dump data success");
    return;
}

void ModelProcess::OutputModelResult()
{
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        // get model output data
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = nullptr;
        aclError ret = ACL_SUCCESS;
        float *outData = nullptr;
        if (!g_isDevice) {
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len, static_cast<int32_t>(ret));
                return;
            }

            // if app is running in host, need copy model output data from device to host
            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }

            outData = reinterpret_cast<float*>(outHostData);
        } else {
            outData = reinterpret_cast<float*>(data);
        }
        map<float, unsigned int, greater<float> > resultMap;
        for (unsigned int j = 0; j < len / sizeof(float); ++j) {
            resultMap[*outData] = j;
            outData++;
        }

        int cnt = 0;
        for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
            // print top 5
            if (++cnt > 5) {
                break;
            }

            INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);
        }
        if (!g_isDevice) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }

    INFO_LOG("output data success");
    return;
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success");
}

Result ModelProcess::Execute()
{
    constexpr int epoch = 1000;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < epoch; i++) {
        aclError ret = aclmdlExecute(modelId_, input_, output_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
                modelId_, static_cast<int32_t>(ret));
            return FAILED;
        }
    }
    auto finish = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    runtime /= epoch;
    std::cerr << "\033[1;31mRuntime: " << runtime / 1e6 << "s\033[0m"
            << std::endl;
    std::cerr << "\033[1;31mFrame: " << 1e6 / runtime << "fps\033[0m"
            << std::endl;

    INFO_LOG("model execute success");
    return SUCCESS;
}

void ModelProcess::UnloadModel()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelWorkPtr_ != nullptr) {
        (void)aclrtFree(modelWorkPtr_);
        modelWorkPtr_ = nullptr;
        modelWorkSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        (void)aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
    modelId_ = 0;
}
