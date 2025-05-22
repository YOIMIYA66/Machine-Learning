// app.js - 前端交互逻辑

// 全局状态变量
let modelTooltipElement = null; // 用于模型描述提示框
let currentData = {
    path: null,
    fileName: null,
    columns: [],
    columnTypes: {},
    rowCount: 0,
    columnCount: 0,
    preview: [],
    analysisCompleted: false,
};

// 默认数据集路径
const DEFAULT_DATASETS = [
    'c:\\Users\\86198\\Desktop\\Study\\机器学习\\Machine Learning\\北京市空气质量数据.xlsx',
    'c:\\Users\\86198\\Desktop\\Study\\机器学习\\Machine Learning\\air_data.csv',
    'c:\\Users\\86198\\Desktop\\Study\\机器学习\\Machine Learning\\离婚诉讼文本.json'
];
let selectedModelName = null;
let selectedTargetColumn = null;
let activeCharts = {}; // 用于存储Chart.js实例，方便销毁和更新
const toastTimeouts = {}; // 存储toast的超时ID

// API端点常量 (根据您的后端调整)
const API_ENDPOINTS = {
    UPLOAD: '/api/ml/upload',
    ANALYZE: '/api/ml/analyze',
    MODELS: '/api/ml/models', // Fetches all models for various selectors
    CHAT: '/api/chat',
    // 移除高级工具相关的API端点
};

// 模型类别分组，便于前端展示和选择 (与后端 ml_models.py 保持一致)
const MODEL_CATEGORIES = {
    "regression": ["linear_regression", "random_forest_regressor"],
    "classification": ["logistic_regression", "knn_classifier", "decision_tree", "svm_classifier", "naive_bayes", "random_forest_classifier"],
    "clustering": ["kmeans"],
    "ensemble": ["voting_classifier", "voting_regressor", "stacking_classifier", "stacking_regressor", "bagging_classifier", "bagging_regressor"]
};

// 固定模型详细信息 - 用于数据与模型页面
const FIXED_MODEL_DETAILS = {
    "linear_regression": {
        "internal_name": "linear_regression",
        "display_name": "线性回归模型",
        "icon_class": "fa-chart-line",
        "description": "用于连续变量预测的基本线性模型。"
    },
    "logistic_regression": {
        "internal_name": "logistic_regression",
        "display_name": "逻辑回归模型",
        "icon_class": "fa-code-branch",
        "description": "用于二分类问题的概率模型。"
    },
    "knn_classifier": {
        "internal_name": "knn_classifier",
        "display_name": "K-近邻法预测模型(KNN)",
        "icon_class": "fa-project-diagram",
        "description": "基于最近邻样本进行分类或回归的算法。"
    },
    "decision_tree": {
        "internal_name": "decision_tree",
        "display_name": "决策树",
        "icon_class": "fa-sitemap",
        "description": "使用树形结构进行决策的分类模型。"
    },
    "svm_classifier": {
        "internal_name": "svm_classifier",
        "display_name": "向量机模型",
        "icon_class": "fa-vector-square",
        "description": "通过最优超平面进行分类的算法。"
    },
    "naive_bayes": {
        "internal_name": "naive_bayes",
        "display_name": "朴素贝叶斯分类器",
        "icon_class": "fa-percentage",
        "description": "基于贝叶斯定理的快速分类器。"
    },
    "kmeans": {
        "internal_name": "kmeans",
        "display_name": "K-Means 模型",
        "icon_class": "fa-object-group",
        "description": "将数据分成K个簇的聚类算法。"
    }
};

// 移除高级工具页面相关的动态获取模型函数
// ... existing code ...

// 根据模型内部名称获取其所属类别键名
function getCategoryForModel(modelInternalName) {
    // 此处MODEL_CATEGORIES应与后端ml_models.py中的定义保持一致或从后端获取
    // 为简化，这里直接使用前端已有的MODEL_CATEGORIES
    for (const category in MODEL_CATEGORIES) {
        if (MODEL_CATEGORIES[category].includes(modelInternalName)) {
            return category;
        }
    }
    // Fallback for models not explicitly in MODEL_CATEGORIES but in FIXED_MODEL_DETAILS
    const modelToCategoryMap = {
        "linear_regression": "regression",
        "logistic_regression": "classification",
        "knn_classifier": "classification",
        "decision_tree": "classification",
        "svm_classifier": "classification",
        "naive_bayes": "classification",
        "kmeans": "clustering"
    };
    return modelToCategoryMap[modelInternalName] || "other"; // Default to 'other'
}

// ... existing code ...

// DOM元素选择器 (集中管理)
const DOM = {
    tabs: () => document.querySelectorAll('.main-tabs .tab'),
    tabContents: () => document.querySelectorAll('.tab-content-area'),
    uploadDataShortcutBtn: () => document.getElementById('uploadDataShortcutBtn'),
    toggleUploadBtn: () => document.getElementById('toggleUploadBtn'),
    uploadContainer: () => document.getElementById('uploadContainer'),
    uploadForm: () => document.getElementById('uploadForm'),
    dataFile: () => document.getElementById('dataFile'),
    analyzeDataBtn: () => document.getElementById('analyzeDataBtn'),
    dataPreview: () => document.getElementById('dataPreview'),
    dataAnalysisResults: () => document.getElementById('dataAnalysisResults'),
    rowCount: () => document.getElementById('rowCount'),
    columnCount: () => document.getElementById('columnCount'),
    recommendedModels: () => document.getElementById('recommendedModels'),
    targetColumnSelector: () => document.getElementById('targetColumnSelector'),
    modelGrid: () => document.getElementById('modelGrid'),
    modelGridPlaceholder: () => document.querySelector('.model-grid-placeholder'),
    modelCountBadge: () => document.getElementById('modelCountBadge'),
    selectedModelInfo: () => document.getElementById('selectedModelInfo'),
    queryModeSelector: () => document.querySelectorAll('input[name="queryMode"]'),
    queryInput: () => document.getElementById('queryInput'),
    queryInputLabel: () => document.getElementById('queryInputLabel'),
    submitQueryButton: () => document.getElementById('submitQueryButton'),
    submitQueryIcon: () => document.getElementById('submitQueryIcon'),
    submitQueryText: () => document.getElementById('submitQueryText'),
    modeSpecificInfo: () => document.getElementById('modeSpecificInfo'),
    exampleQueryList: () => document.getElementById('exampleQueryList'),
    toastContainer: () => document.getElementById('toast-container'),
    loadingSpinnerContainer: () => document.getElementById('loadingSpinnerContainer'),
    responseSection: () => document.getElementById('responseSection'),
    responseText: () => document.getElementById('responseText'),
    visualizationDisplayArea: () => document.getElementById('visualizationDisplayArea'),
    sourceDocumentsArea: () => document.getElementById('sourceDocumentsArea'),
    sourceDocumentsList: () => document.getElementById('sourceDocumentsList'),
    sourceDocumentsMessage: () => document.getElementById('sourceDocumentsMessage'),
    vizTabs: () => document.querySelectorAll('.result-viz-tabs .tab'),
    vizContents: () => document.querySelectorAll('.viz-content-panel'),
    mainResultChart: () => document.getElementById('mainResultChart'),
    mainChartMessage: () => document.getElementById('mainChartMessage'),
    featureImportanceChart: () => document.getElementById('featureImportanceChart'),
    // 移除高级工具相关的DOM元素选择器
};

// ... existing code ...

async function main() { // Make main async
    initTabs();
    initUploadToggle();
    initUploadForm();
    await loadAvailableModels(); // 加载可用的模型
    initModelSelectionDelegation();
    updateQueryInputState();
    initQuerySubmission();
    initVisualizationTabs();
    initExampleQueries();
    initDataUploadShortcut();
    initParticlesJS();
    // 移除高级工具初始化
    
    console.log("应用程序初始化完成!");
}

// ... existing code ...

async function loadAvailableModels() {
    try {
        // Display loading placeholder
        DOM.modelGrid().innerHTML = '<div class="w-full text-center py-8"><span class="loading loading-dots loading-lg"></span><p class="text-muted mt-3">正在加载模型...</p></div>';
        
        // Fetch models from the server
        const response = await fetch(API_ENDPOINTS.MODELS);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`Error fetching models: ${data.error || response.statusText}`);
        }
        
        // Process and display models
        const models = data.models || [];
        DOM.modelCountBadge().textContent = `已加载 ${models.length} 个模型`;
        
        // 准备展示的模型数据
        const modelsToDisplay = models.map(model => {
            // 为每个模型补充显示信息
            const modelType = model.type || "unknown";
            const internalName = model.internal_name || modelType;
            
            // 尝试从FIXED_MODEL_DETAILS获取详细信息
            const details = FIXED_MODEL_DETAILS[internalName] || {
                display_name: model.display_name || getModelDisplayName(internalName),
                icon_class: model.icon_class || getDefaultModelIcon(internalName),
                description: model.description || `${getModelDisplayName(internalName)} 模型`
            };
            
            return {
                internal_name: internalName,
                display_name: details.display_name,
                type: modelType,
                description: details.description,
                icon_class: details.icon_class,
                name: model.name || details.display_name,
                category: getCategoryForModel(internalName)
            };
        });
        
        // 按类别组织模型
        const modelsByCategory = {};
        
        modelsToDisplay.forEach(model => {
            const category = model.category;
            if (!modelsByCategory[category]) {
                modelsByCategory[category] = [];
            }
            modelsByCategory[category].push(model);
        });
        
        // 构建模型网格
        let gridHTML = '';
        
        // 按类别顺序展示
        const categories = Object.keys(modelsByCategory).sort((a, b) => {
            // 自定义排序顺序
            const order = ["regression", "classification", "clustering", "ensemble", "other"];
            return order.indexOf(a) - order.indexOf(b);
        });
        
        categories.forEach(category => {
            const models = modelsByCategory[category];
            
            gridHTML += `
                <div class="model-category mb-6">
                    <h4 class="text-lg font-medium mb-3 flex items-center">
                        <span class="badge badge-sm border-transparent bg-primary-hex/20 text-primary-hex py-1 px-2 mr-2">${getCategoryDisplayName(category)}</span>
                        <span class="text-sm font-normal text-muted">${models.length} 个模型</span>
                    </h4>
                    <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        ${models.map(model => createModelCardElement(
                            model.internal_name,
                            model.display_name,
                            model.type,
                            model.description,
                            model.icon_class
                        )).join('')}
                    </div>
                </div>
            `;
        });
        
        // 更新模型网格
        DOM.modelGrid().innerHTML = gridHTML || '<p class="text-center text-muted py-8">暂无可用模型</p>';
        
        // 移除与高级工具相关的代码
    } catch (error) {
        console.error("Failed to load models:", error);
        DOM.modelGrid().innerHTML = `<div class="alert alert-error shadow-lg"><div><i class="fas fa-exclamation-circle"></i><span>加载模型失败: ${error.message}</span></div></div>`;
    }
}

// ... existing code ...

// 移除高级工具相关函数，包括:
// async function initAdvancedTools()
// function loadDefaultDatasets()
// async function populateAdvancedToolSelectors(modelsList)
// function populateModelSelector(selectEl, models, placeholder)
// function initModelVersioning()
// async function fetchAndDisplayModelVersions(modelName)
// function initModelComparison()
// function loadDefaultDatasets()
// function initializeModelSelectors()
// function populateSelectWithOptions(selectElement, options, placeholderText = "请选择")
// function formatMetricName(metric)
// function initEnsembleBuilding()
// function addDynamicModelSelector(container, placeholderEl, modelsList, currentCount, maxCount, selectClass, labelPrefix, updateCountCallback)
// function initModelDeployment()
// async function fetchAndDisplayDeployedModels()

// ... 其他辅助工具函数 ...

document.addEventListener('DOMContentLoaded', main);