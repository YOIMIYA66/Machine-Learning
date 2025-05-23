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
// 添加学习路径相关状态
let currentLearningPath = null;
let learningPathCharts = {};

// API端点常量 (根据您的后端调整)
const API_ENDPOINTS = {
    UPLOAD: '/api/ml/upload',
    ANALYZE: '/api/ml/analyze',
    MODELS: '/api/ml/models', // Fetches all models for various selectors
    CHAT: '/api/chat',
    QUERY: '/query', // 添加查询端点
    // 添加学习路径相关API端点
    LEARNING_PATH_CREATE: '/api/learning_path/create',
    LEARNING_PATH_USER: '/api/learning_path/user/',
    LEARNING_PATH_UPDATE: '/api/learning_path/update_progress',
    LEARNING_PATH_PREDICT_MASTERY: '/api/learning_path/predict/mastery',
    LEARNING_PATH_PREDICT_COMPLETION: '/api/learning_path/predict/completion_time',
    // 移除高级工具相关的API端点
};

// // 模型类别分组，便于前端展示和选择 (与后端 ml_models.py 保持一致)
// const MODEL_CATEGORIES = {
//     "regression": ["linear_regression", "random_forest_regressor"],
//     "classification": ["logistic_regression", "knn_classifier", "decision_tree", "svm_classifier", "naive_bayes", "random_forest_classifier"],
//     "clustering": ["kmeans"],
//     "ensemble": ["voting_classifier", "voting_regressor", "stacking_classifier", "stacking_regressor", "bagging_classifier", "bagging_regressor"]
// };

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
    // 添加查询响应容器方法
    queryResponseContainer: () => {
        // 直接返回HTML中已存在的容器
        return document.getElementById('queryResponseContainer');
    },
    // 移除高级工具相关的DOM元素选择器
};

/**
 * 检查系统状态和关键DOM元素
 */
function checkSystemStatus() {
    const issues = [];
    
    // 检查主要标签页元素
    const requiredTabs = [
        'tab-link-dialogue', 'tab-link-learningPath', 'tab-link-dataUpload', 
        'tab-link-techLab', 'tab-link-results'
    ];
    
    requiredTabs.forEach(tabId => {
        if (!document.getElementById(tabId)) {
            issues.push(`主标签页缺失: ${tabId}`);
        }
    });
    
    // 检查学习路径相关元素
    const learningPathElements = [
        'pathTitle', 'pathDescription', 'overallProgress', 'progressDesc',
        'estimatedCompletionTime', 'completionTimeDesc', 'masteryProbability',
        'masteryDesc', 'weeklyStudyHoursSlider', 'weeklyStudyHoursValue'
    ];
    
    learningPathElements.forEach(elementId => {
        if (!document.getElementById(elementId)) {
            issues.push(`学习路径元素缺失: ${elementId}`);
        }
    });
    
    // 检查技术实验室元素
    const techLabElements = [
        'runSimulationBtn', 'resetSimulationBtn', 'modelComparisonChart',
        'metricsTableBody', 'simulationAnalysisText'
    ];
    
    techLabElements.forEach(elementId => {
        if (!document.getElementById(elementId)) {
            issues.push(`技术实验室元素缺失: ${elementId}`);
        }
    });
    
    // 检查查询相关元素
    const queryElements = [
        'queryInput', 'submitQueryButton', 'queryInputLabel', 'modeSpecificInfo'
    ];
    
    queryElements.forEach(elementId => {
        if (!document.getElementById(elementId)) {
            issues.push(`查询元素缺失: ${elementId}`);
        }
    });
    
    // 检查数据上传相关元素
    const uploadElements = [
        'uploadForm', 'dataFile', 'analyzeDataBtn', 'dataPreview', 'modelGrid'
    ];
    
    uploadElements.forEach(elementId => {
        if (!document.getElementById(elementId)) {
            issues.push(`数据上传元素缺失: ${elementId}`);
        }
    });
    
    // 检查外部库
    if (typeof Chart === 'undefined') {
        issues.push('Chart.js 库未加载');
    }
    
    if (typeof marked === 'undefined') {
        console.warn('marked 库未加载，Markdown解析将使用简单文本替换');
    }
    
    if (typeof particlesJS === 'undefined') {
        console.warn('particles.js 库未加载，背景粒子效果将被跳过');
    }
    
    // 报告问题
    if (issues.length > 0) {
        console.error('系统状态检查发现以下问题:');
        issues.forEach(issue => console.error(`- ${issue}`));
        
        // 显示用户友好的错误信息
        showToast('系统检查', `发现 ${issues.length} 个配置问题，部分功能可能受影响`, 'warning', 8000);
        
        return false;
    } else {
        console.log('✅ 系统状态检查通过');
        return true;
    }
}

async function main() { // Make main async
    try {
        console.log("🚀 启航者 AI - 开始初始化应用程序...");
        
        // 首先执行系统状态检查
        const systemOK = checkSystemStatus();
        if (!systemOK) {
            console.warn("系统状态检查发现问题，但继续初始化...");
        }
        
    initTabs();
        console.log("✅ 标签页初始化完成");
        
    initUploadToggle();
        console.log("✅ 上传切换初始化完成");
        
    initUploadForm();
        console.log("✅ 上传表单初始化完成");
        
    await loadAvailableModels(); // 加载可用的模型
        console.log("✅ 模型加载完成");
        
    initModelSelectionDelegation();
        console.log("✅ 模型选择初始化完成");
        
    updateQueryInputState();
        console.log("✅ 查询输入状态更新完成");
        
    initQuerySubmission();
        console.log("✅ 查询提交初始化完成");
        
    initVisualizationTabs();
        console.log("✅ 可视化标签页初始化完成");
        
    initExampleQueries();
        console.log("✅ 示例查询初始化完成");
        
    initDataUploadShortcut();
        console.log("✅ 数据上传快捷方式初始化完成");
        
    initParticlesJS();
        console.log("✅ 粒子背景初始化完成");
        
    // 初始化学习路径相关功能
    initLearningPathFeatures();
        console.log("✅ 学习路径功能初始化完成");
        
        // 初始化技术实验室功能
        initTechLabFeatures();
        console.log("✅ 技术实验室功能初始化完成");
    
        console.log("🎉 应用程序初始化完成!");
        
        // 显示初始化完成提示
        showToast('系统就绪', '启航者 AI 已准备就绪，开始您的学习之旅！', 'success', 3000);
    } catch (error) {
        console.error("❌ 应用程序初始化失败:", error);
        showToast('初始化错误', `应用程序初始化失败: ${error.message}`, 'error', 10000);
    }
}

/**
 * 初始化标签页切换功能
 * 为主标签页添加点击事件监听器
 */
function initTabs() {
    const tabs = DOM.tabs();
    const tabContents = DOM.tabContents();
    
    if (!tabs || !tabContents) {
        console.error("找不到标签页或内容元素");
        return;
    }
    
    // 为每个标签页添加点击事件监听器
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // 获取目标标签页ID
            const targetTabId = tab.getAttribute('data-tab');
            if (!targetTabId) return;
            
            // 移除所有标签页的active类
            tabs.forEach(t => t.classList.remove('tab-active'));
            
            // 为当前点击的标签页添加active类
            tab.classList.add('tab-active');
            
            // 隐藏所有内容区域
            tabContents.forEach(content => content.classList.add('hidden'));
            
            // 显示对应的内容区域
            const targetContent = document.getElementById(`tab-content-${targetTabId}`);
            if (targetContent) {
                targetContent.classList.remove('hidden');
            }
            
            // 更新ARIA属性
            tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
            tab.setAttribute('aria-selected', 'true');
            
            // 如果切换到学习路径标签页，加载用户学习路径
            if (targetTabId === 'learningPath') {
                if (typeof loadUserLearningPaths === 'function') {
                    loadUserLearningPaths();
                }
            }
        });
    });
}

/**
 * 初始化可视化标签页切换
 */
function initVisualizationTabs() {
    const vizTabs = DOM.vizTabs();
    const vizContents = DOM.vizContents();
    
    if (!vizTabs || !vizContents) return;
    
    vizTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // 如果标签页被禁用，则不处理
            if (tab.classList.contains('disabled-tab')) return;
            
            // 获取目标可视化ID
            const targetVizId = tab.getAttribute('data-viz');
            if (!targetVizId) return;
            
            // 移除所有标签页的active类
            vizTabs.forEach(t => t.classList.remove('tab-active'));
            
            // 为当前点击的标签页添加active类
            tab.classList.add('tab-active');
            
            // 隐藏所有内容区域
            vizContents.forEach(content => content.classList.add('hidden'));
            
            // 显示对应的内容区域
            const targetContent = document.getElementById(`viz-content-${targetVizId}`);
            if (targetContent) {
                targetContent.classList.remove('hidden');
            }
            
            // 更新ARIA属性
            vizTabs.forEach(t => t.setAttribute('aria-selected', 'false'));
            tab.setAttribute('aria-selected', 'true');
        });
    });
}

/**
 * 初始化数据上传切换按钮
 */
function initUploadToggle() {
    const toggleBtn = DOM.toggleUploadBtn();
    const uploadContainer = DOM.uploadContainer();
    
    if (!toggleBtn || !uploadContainer) return;
    
    toggleBtn.addEventListener('click', () => {
        // 切换显示/隐藏上传容器
        const isHidden = uploadContainer.classList.contains('hidden');
        
        if (isHidden) {
            // 显示上传容器
            uploadContainer.classList.remove('hidden');
            toggleBtn.innerHTML = '<i class="fas fa-times"></i> 关闭上传';
            toggleBtn.setAttribute('aria-expanded', 'true');
        } else {
            // 隐藏上传容器
            uploadContainer.classList.add('hidden');
            toggleBtn.innerHTML = '<i class="fas fa-file-upload"></i> 上传新数据';
            toggleBtn.setAttribute('aria-expanded', 'false');
        }
    });
}

/**
 * 初始化数据上传快捷按钮
 */
function initDataUploadShortcut() {
    const shortcutBtn = DOM.uploadDataShortcutBtn();
    
    if (!shortcutBtn) return;
    
    shortcutBtn.addEventListener('click', () => {
        // 切换到数据上传标签页
        const dataUploadTab = document.getElementById('tab-link-dataUpload');
        if (dataUploadTab) {
            dataUploadTab.click();
            
            // 显示上传容器
            const uploadContainer = DOM.uploadContainer();
            const toggleBtn = DOM.toggleUploadBtn();
            
            if (uploadContainer && toggleBtn) {
                uploadContainer.classList.remove('hidden');
                toggleBtn.innerHTML = '<i class="fas fa-times"></i> 关闭上传';
                toggleBtn.setAttribute('aria-expanded', 'true');
            }
        }
    });
}

/**
 * 初始化粒子背景
 */
function initParticlesJS() {
    if (typeof particlesJS === 'function') {
        particlesJS('particles-js-bg', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: ['#0284C7', '#06B6D4', '#10B981'] },
                shape: { type: 'circle' },
                opacity: { value: 0.5, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#0284C7',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: true,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'grab' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                },
                modes: {
                    grab: { distance: 140, line_linked: { opacity: 1 } },
                    push: { particles_nb: 4 }
                }
            },
            retina_detect: true
        });
    }
}

// 固定模型列表 - 不再从后端加载
const FIXED_MODELS = [
    {
        internal_name: "linear_regression",
        display_name: "线性回归模型",
        type: "regression", 
        description: "用于连续变量预测的基本线性模型，适合线性关系数据。",
        icon_class: "fa-chart-line",
        category: "regression"
    },
    {
        internal_name: "logistic_regression", 
        display_name: "逻辑回归模型",
        type: "classification",
        description: "用于二分类问题的概率模型，输出概率值。",
        icon_class: "fa-code-branch",
        category: "classification"
    },
    {
        internal_name: "decision_tree",
        display_name: "决策树",
        type: "classification",
        description: "使用树形结构进行决策的分类模型，可解释性强。",
        icon_class: "fa-sitemap", 
        category: "classification"
    },
    {
        internal_name: "random_forest",
        display_name: "随机森林", 
        type: "ensemble",
        description: "集成多棵决策树的强大模型，精度高且抗过拟合。",
        icon_class: "fa-tree",
        category: "ensemble"
    },
    {
        internal_name: "svm_classifier",
        display_name: "支持向量机",
        type: "classification", 
        description: "通过最优超平面进行分类的高效算法。",
        icon_class: "fa-vector-square",
        category: "classification"
    },
    {
        internal_name: "naive_bayes",
        display_name: "朴素贝叶斯分类器",
        type: "classification",
        description: "基于贝叶斯定理的快速分类器，适合文本分类。",
        icon_class: "fa-percentage", 
        category: "classification"
    },
    {
        internal_name: "kmeans",
        display_name: "K-Means聚类",
        type: "clustering",
        description: "将数据分成K个簇的无监督聚类算法。", 
        icon_class: "fa-object-group",
        category: "clustering"
    }
];

async function loadAvailableModels() {
    try {
        console.log("📦 加载固定模型列表...");
        
        // 显示加载状态
        DOM.modelGrid().innerHTML = '<div class="w-full text-center py-8"><span class="loading loading-dots loading-lg"></span><p class="text-muted mt-3">正在加载模型...</p></div>';
        
        // 模拟短暂加载时间以提供更好的用户体验
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // 更新模型计数
        DOM.modelCountBadge().textContent = `已加载 ${FIXED_MODELS.length} 个模型`;
        
        // 按类别组织模型
        const modelsByCategory = {};
        FIXED_MODELS.forEach(model => {
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
            const order = ["regression", "classification", "ensemble", "clustering"];
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
        
        console.log("✅ 固定模型加载完成");
        
    } catch (error) {
        console.error("❌ 加载模型失败:", error);
        DOM.modelGrid().innerHTML = `<div class="alert alert-error shadow-lg"><div><i class="fas fa-exclamation-circle"></i><span>加载模型失败: ${error.message}</span></div></div>`;
    }
}

// 添加学习路径相关功能
function initLearningPathFeatures() {
    // 获取DOM元素
    const createPathBtn = document.getElementById('createPathBtn');
    const editPathBtn = document.getElementById('editPathBtn');
    const refreshPredictionsBtn = document.getElementById('refreshPredictionsBtn');
    const weeklyStudyHoursSlider = document.getElementById('weeklyStudyHoursSlider');
    const weeklyStudyHoursValue = document.getElementById('weeklyStudyHoursValue');
    const updateLearningParamsBtn = document.getElementById('updateLearningParamsBtn');
    
    // 初始化学习路径内部标签页切换
    initLearningPathTabs();
    
    // 初始绑定事件监听器
    if (createPathBtn) {
        createPathBtn.addEventListener('click', () => {
            // 切换到学习导航标签页
            document.getElementById('tab-link-dialogue').click();
            
            // 在查询输入框中添加创建学习路径的提示
            DOM.queryInput().value = "我想学习机器学习，我目前没有相关背景，每周可以学习10小时左右，帮我制定一个学习路径。";
            
            // 聚焦输入框
            DOM.queryInput().focus();
        });
    }
    
    if (editPathBtn) {
        editPathBtn.addEventListener('click', () => {
            showToast('编辑功能', '学习路径编辑功能即将推出，敬请期待！', 'info');
        });
    }
    
    if (refreshPredictionsBtn) {
        refreshPredictionsBtn.addEventListener('click', () => {
            if (currentLearningPath) {
                updateLearningPathPredictions(currentLearningPath.path_id, parseInt(weeklyStudyHoursSlider.value));
            } else {
                showToast('错误', '没有可更新的学习路径', 'error');
            }
        });
    }
    
    // 初始化学习时间滑块
    if (weeklyStudyHoursSlider && weeklyStudyHoursValue) {
        weeklyStudyHoursSlider.addEventListener('input', () => {
            weeklyStudyHoursValue.textContent = `${weeklyStudyHoursSlider.value}小时/周`;
        });
    }
    
    if (updateLearningParamsBtn) {
        updateLearningParamsBtn.addEventListener('click', () => {
            if (currentLearningPath) {
                const weeklyHours = parseInt(weeklyStudyHoursSlider.value);
                updateLearningPathPredictions(currentLearningPath.path_id, weeklyHours);
                showToast('成功', `已更新学习参数为每周${weeklyHours}小时`, 'success');
            } else {
                showToast('错误', '没有可更新的学习路径', 'error');
            }
        });
    }
    
    // 学习路径标签页点击时加载用户学习路径
    document.getElementById('tab-link-learningPath').addEventListener('click', () => {
        loadUserLearningPaths();
    });
}

/**
 * 初始化学习路径内部标签页
 */
function initLearningPathTabs() {
    // 绑定标签页点击事件
    const tabModules = document.getElementById('tab-modules');
    const tabKnowledge = document.getElementById('tab-knowledge');
    const tabProgress = document.getElementById('tab-progress');
    
    if (tabModules) {
        tabModules.addEventListener('click', () => {
            switchLearningPathTab('modules');
        });
    }
    
    if (tabKnowledge) {
        tabKnowledge.addEventListener('click', () => {
            switchLearningPathTab('knowledge');
        });
    }
    
    if (tabProgress) {
        tabProgress.addEventListener('click', () => {
            switchLearningPathTab('progress');
        });
    }
    
    // 绑定生成知识库内容按钮
    const generateKnowledgeBtn = document.getElementById('generateKnowledgeBtn');
    if (generateKnowledgeBtn) {
        generateKnowledgeBtn.addEventListener('click', generateKnowledgeContent);
    }
}

/**
 * 切换学习路径标签页
 */
function switchLearningPathTab(tabName) {
    // 更新标签页状态
    document.querySelectorAll('#tab-content-learningPath .tab').forEach(tab => {
        tab.classList.remove('tab-active');
    });
    
    document.getElementById(`tab-${tabName}`).classList.add('tab-active');
    
    // 显示对应内容
    document.querySelectorAll('#tab-content-learningPath .tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    const targetContent = document.getElementById(`content-${tabName}`);
    if (targetContent) {
        targetContent.classList.remove('hidden');
        
        // 如果切换到进度分析，更新图表
        if (tabName === 'progress' && currentLearningPath) {
            updateLearningPathPredictions(currentLearningPath.path_id, currentLearningPath.weekly_hours);
        }
    }
}

/**
 * 生成AI知识库内容
 */
async function generateKnowledgeContent() {
    try {
        if (!currentLearningPath) {
            showToast('错误', '请先创建学习路径', 'error');
            return;
        }
        
        const generateBtn = document.getElementById('generateKnowledgeBtn');
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 生成中...';
        
        // 准备请求数据
        const requestData = {
            query: `请根据我的学习路径"${currentLearningPath.goal}"生成相关的学习资料和知识内容`,
            mode: 'general_llm',
            learning_path_context: {
                goal: currentLearningPath.goal,
                modules: currentLearningPath.modules?.slice(0, 5), // 发送前5个模块
                progress: currentLearningPath.progress_percentage || 0
            },
            content_type: 'knowledge_base'
        };
        
        // 调用查询API
        const response = await fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '生成内容失败');
        }
        
        // 显示生成的内容
        displayKnowledgeContent(data.answer);
        
        showToast('成功', 'AI知识内容生成完成', 'success');
        
    } catch (error) {
        console.error('生成知识内容失败:', error);
        showToast('错误', `生成知识内容失败: ${error.message}`, 'error');
    } finally {
        // 恢复按钮状态
        const generateBtn = document.getElementById('generateKnowledgeBtn');
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic mr-2"></i>生成个性化内容';
    }
}

/**
 * 显示知识库内容
 */
function displayKnowledgeContent(content) {
    const articlesContainer = document.getElementById('knowledgeArticles');
    if (!articlesContainer) return;
    
    // 清空现有内容
    articlesContainer.innerHTML = '';
    
    // 创建知识内容卡片
    const contentCard = document.createElement('div');
    contentCard.className = 'col-span-full p-6 border border-base-300 rounded-lg bg-base-100 prose prose-sm max-w-none';
    contentCard.innerHTML = marked ? marked.parse(content) : content.replace(/\n/g, '<br>');
    
    articlesContainer.appendChild(contentCard);
    
    // 添加一些示例知识卡片
    const exampleCards = [
        {
            title: '📚 推荐阅读',
            content: '基于您的学习进度，推荐以下资料进行深入学习...',
            icon: 'fa-book'
        },
        {
            title: '💡 学习技巧',
            content: 'AI为您推荐的高效学习方法和记忆技巧...',
            icon: 'fa-lightbulb'
        },
        {
            title: '🔗 相关资源',
            content: '与您当前学习模块相关的在线资源和工具...',
            icon: 'fa-link'
        }
    ];
    
    exampleCards.forEach(card => {
        const cardElement = document.createElement('div');
        cardElement.className = 'p-4 border border-base-300 rounded-lg bg-base-100 hover:shadow-md transition-shadow';
        cardElement.innerHTML = `
            <div class="flex items-center mb-3">
                <i class="fas ${card.icon} text-primary-hex mr-2"></i>
                <h6 class="font-medium">${card.title}</h6>
            </div>
            <p class="text-sm text-muted">${card.content}</p>
            <button class="btn btn-xs btn-outline mt-3">查看详情</button>
        `;
        articlesContainer.appendChild(cardElement);
    });
}

/**
 * 加载用户的学习路径
 */
async function loadUserLearningPaths() {
    try {
        // 使用默认用户ID - 在实际应用中应该从登录系统获取
        const userId = 'default_user';
        
        // 显示加载状态
        document.getElementById('learningPathContent').classList.add('hidden');
        document.getElementById('emptyLearningPathMessage').innerHTML = `
            <div class="text-center py-8">
                <span class="loading loading-spinner loading-lg text-primary-hex"></span>
                <p class="text-muted mt-3">正在加载学习路径...</p>
            </div>
        `;
        document.getElementById('emptyLearningPathMessage').classList.remove('hidden');
        
        // 调用API获取用户的学习路径
        const response = await fetch(`${API_ENDPOINTS.LEARNING_PATH_USER}${userId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '加载学习路径失败');
        }
        
        // 检查是否有学习路径
        if (data.success && data.paths && data.paths.length > 0) {
            // 保存当前学习路径
            currentLearningPath = data.paths[0]; // 取最新的一条
            
            // 显示学习路径内容
            document.getElementById('emptyLearningPathMessage').classList.add('hidden');
            document.getElementById('learningPathContent').classList.remove('hidden');
            
            // 渲染学习路径详情
            renderLearningPathDetails(currentLearningPath);
            
            // 更新预测数据
            updateLearningPathPredictions(currentLearningPath.path_id, currentLearningPath.weekly_hours);
        } else {
            // 显示创建学习路径提示
            document.getElementById('learningPathContent').classList.add('hidden');
            document.getElementById('emptyLearningPathMessage').innerHTML = `
                <i class="fas fa-route text-6xl text-muted mb-4 opacity-50"></i>
                <p class="text-muted">您尚未创建学习路径。请在"学习导航"标签页与AI对话，设定您的学习目标。</p>
                <button id="createPathBtn" class="btn btn-primary mt-4">
                    <i class="fas fa-plus-circle mr-2"></i>创建学习路径
                </button>
            `;
            document.getElementById('emptyLearningPathMessage').classList.remove('hidden');
            
            // 重新绑定创建按钮事件
            document.getElementById('createPathBtn').addEventListener('click', () => {
                document.getElementById('tab-link-dialogue').click();
                DOM.queryInput().value = "我想学习机器学习，我目前没有相关背景，每周可以学习10小时左右，帮我制定一个学习路径。";
                DOM.queryInput().focus();
            });
        }
    } catch (error) {
        console.error('加载学习路径失败:', error);
        showToast('错误', `加载学习路径失败: ${error.message}`, 'error');
        
        // 显示错误提示
        document.getElementById('learningPathContent').classList.add('hidden');
        document.getElementById('emptyLearningPathMessage').innerHTML = `
            <i class="fas fa-exclamation-triangle text-6xl text-error mb-4 opacity-50"></i>
            <p class="text-muted">加载学习路径失败。${error.message}</p>
            <button id="retryLoadPathBtn" class="btn btn-primary mt-4">
                <i class="fas fa-redo mr-2"></i>重试
            </button>
        `;
        document.getElementById('emptyLearningPathMessage').classList.remove('hidden');
        
        // 绑定重试按钮事件
        document.getElementById('retryLoadPathBtn').addEventListener('click', loadUserLearningPaths);
    }
}

/**
 * 渲染学习路径详情
 * @param {Object} path 学习路径对象
 */
function renderLearningPathDetails(path) {
    if (!path) {
        console.error('renderLearningPathDetails: path参数为空');
        return;
    }
    
    // 更新标题和描述，添加null检查
    const pathTitle = document.getElementById('pathTitle');
    const pathDescription = document.getElementById('pathDescription');
    
    if (pathTitle) {
        pathTitle.textContent = path.goal || '个性化学习路径';
    } else {
        console.warn('pathTitle元素不存在');
    }
    
    if (pathDescription) {
        pathDescription.textContent = `基于您的背景和目标设计的学习路径，包含${path.total_modules || 0}个模块`;
    } else {
        console.warn('pathDescription元素不存在');
    }
    
    // 更新进度信息
    const progressPercentage = path.progress_percentage || 0;
    const overallProgress = document.getElementById('overallProgress');
    const progressDesc = document.getElementById('progressDesc');
    
    if (overallProgress) {
        overallProgress.textContent = `${progressPercentage}%`;
    } else {
        console.warn('overallProgress元素不存在');
    }
    
    const completedCount = (path.completed_modules || []).length;
    const totalCount = path.total_modules || 0;
    if (progressDesc) {
        progressDesc.textContent = `已完成${completedCount}/${totalCount}个模块`;
    } else {
        console.warn('progressDesc元素不存在');
    }
    
    // 更新估计完成时间
    const estimatedHours = path.estimated_total_hours || 0;
    const estimatedCompletionTime = document.getElementById('estimatedCompletionTime');
    const completionTimeDesc = document.getElementById('completionTimeDesc');
    
    if (estimatedCompletionTime) {
        estimatedCompletionTime.textContent = `${estimatedHours}小时`;
    } else {
        console.warn('estimatedCompletionTime元素不存在');
    }
    
    if (completionTimeDesc) {
        completionTimeDesc.textContent = `基于每周${path.weekly_hours || 10}小时学习强度`;
    } else {
        console.warn('completionTimeDesc元素不存在');
    }
    
    // 更新掌握概率信息
    const masteryProbability = document.getElementById('masteryProbability');
    const masteryDesc = document.getElementById('masteryDesc');
    
    if (masteryProbability) {
        // 使用默认值，如果有实际预测结果会在updateLearningPathPredictions中更新
        masteryProbability.textContent = '计算中...';
    } else {
        console.warn('masteryProbability元素不存在');
    }
    
    if (masteryDesc) {
        masteryDesc.textContent = `基于每周${path.weekly_hours || 10}小时学习强度`;
    } else {
        console.warn('masteryDesc元素不存在');
    }
    
    // 设置滑块默认值
    const weeklyStudyHoursSlider = document.getElementById('weeklyStudyHoursSlider');
    const weeklyStudyHoursValue = document.getElementById('weeklyStudyHoursValue');
    
    if (weeklyStudyHoursSlider) {
        weeklyStudyHoursSlider.value = path.weekly_hours || 10;
    } else {
        console.warn('weeklyStudyHoursSlider元素不存在');
    }
    
    if (weeklyStudyHoursValue) {
        weeklyStudyHoursValue.textContent = `${path.weekly_hours || 10}小时/周`;
    } else {
        console.warn('weeklyStudyHoursValue元素不存在');
    }
    
    // 渲染学习模块列表
    renderLearningModules(path.modules || [], path.completed_modules || [], path.current_module_id);
}

/**
 * 渲染学习模块列表
 * @param {Array} modules 模块列表
 * @param {Array} completedModules 已完成模块ID列表
 * @param {String} currentModuleId 当前模块ID
 */
function renderLearningModules(modules, completedModules, currentModuleId) {
    const modulesContainer = document.getElementById('learningModules');
    if (!modulesContainer) return;
    
    // 清空现有内容
    modulesContainer.innerHTML = '';
    
    // 如果没有模块，显示提示
    if (!modules || modules.length === 0) {
        modulesContainer.innerHTML = `
            <div class="text-center py-6">
                <p class="text-muted">未找到学习模块。</p>
            </div>
        `;
        return;
    }
    
    // 添加每个模块
    modules.forEach((module, index) => {
        const moduleId = module.id;
        const isCompleted = completedModules.includes(moduleId);
        const isCurrent = moduleId === currentModuleId;
        
        // 确定模块状态
        let statusClass = 'neutral-content/30';
        let statusBadge = '<span class="badge badge-outline">未开始</span>';
        
        if (isCompleted) {
            statusClass = 'success';
            statusBadge = '<span class="badge badge-success">已完成</span>';
        } else if (isCurrent) {
            statusClass = 'primary';
            statusBadge = '<span class="badge badge-primary">进行中</span>';
        }
        
        // 创建模块元素
        const moduleElement = document.createElement('div');
        moduleElement.className = 'module-item p-4 border border-base-300 rounded-lg bg-base-100 relative';
        moduleElement.innerHTML = `
            <div class="absolute top-0 left-0 h-full w-1 bg-${statusClass} rounded-l-lg"></div>
            <div class="flex flex-col md:flex-row justify-between">
                <div>
                    <h5 class="font-medium">${index + 1}. ${module.name || '未命名模块'}</h5>
                    <p class="text-sm text-muted mt-1">${module.description || '没有描述'}</p>
                </div>
                <div class="mt-3 md:mt-0 flex flex-col items-end">
                    ${statusBadge}
                    <span class="text-xs text-muted mt-1">预计学习时间: ${module.estimated_hours || 0}小时</span>
                </div>
            </div>
            ${isCurrent && !isCompleted ? `
                <div class="mt-3">
                    <div class="w-full bg-base-200 rounded-full h-2.5">
                        <div class="bg-primary h-2.5 rounded-full" style="width: 50%"></div>
                    </div>
                    <div class="flex justify-between text-xs text-muted mt-1">
                        <span>约50%</span>
                        <span>剩余约${Math.round(module.estimated_hours * 0.5 * 10) / 10}小时</span>
                    </div>
                </div>
            ` : ''}
            ${!isCompleted && !isCurrent ? `
                <div class="mt-3 flex items-center gap-2">
                    <button class="btn btn-xs btn-outline start-module-btn" data-module-id="${moduleId}">开始学习</button>
                    <div class="tooltip" data-tip="预测您掌握此模块的概率">
                        <span class="flex items-center gap-1 text-xs">
                            <i class="fas fa-graduation-cap text-primary-hex"></i>
                            掌握概率: 计算中...
                        </span>
                    </div>
                </div>
            ` : ''}
        `;
        
        // 将模块添加到容器
        modulesContainer.appendChild(moduleElement);
    });
    
    // 绑定开始学习按钮事件
    document.querySelectorAll('.start-module-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const moduleId = btn.getAttribute('data-module-id');
            startLearningModule(moduleId);
        });
    });
}

/**
 * 开始学习模块
 * @param {String} moduleId 模块ID
 */
async function startLearningModule(moduleId) {
    try {
        if (!currentLearningPath) {
            showToast('错误', '没有活动的学习路径', 'error');
            return;
        }
        
        // 显示加载状态
        const btn = document.querySelector(`.start-module-btn[data-module-id="${moduleId}"]`);
        if (btn) {
            btn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 更新中...';
            btn.disabled = true;
        }
        
        // 调用API更新当前模块
        const response = await fetch(API_ENDPOINTS.LEARNING_PATH_UPDATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path_id: currentLearningPath.path_id,
                current_module_id: moduleId
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '更新学习进度失败');
        }
        
        // 更新当前学习路径
        currentLearningPath = data.path;
        
        // 重新渲染学习路径详情
        renderLearningPathDetails(currentLearningPath);
        
        // 显示成功提示
        showToast('成功', '已开始学习新模块', 'success');
    } catch (error) {
        console.error('开始学习模块失败:', error);
        showToast('错误', `开始学习模块失败: ${error.message}`, 'error');
        
        // 恢复按钮状态
        const btn = document.querySelector(`.start-module-btn[data-module-id="${moduleId}"]`);
        if (btn) {
            btn.innerHTML = '开始学习';
            btn.disabled = false;
        }
    }
}

/**
 * 更新学习路径预测
 * @param {String} pathId 学习路径ID
 * @param {Number} weeklyHours 每周学习时间
 */
async function updateLearningPathPredictions(pathId, weeklyHours) {
    try {
        // 默认用户ID
        const userId = 'default_user';
        
        // 显示加载状态
        document.getElementById('masteryProbability').innerHTML = '<span class="loading loading-spinner loading-sm"></span>';
        
        // 调用API获取完成时间预测
        const completionResponse = await fetch(API_ENDPOINTS.LEARNING_PATH_PREDICT_COMPLETION, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                path_id: pathId,
                weekly_hours: weeklyHours
            })
        });
        
        const completionData = await completionResponse.json();
        
        if (!completionResponse.ok) {
            throw new Error(completionData.error || '获取完成时间预测失败');
        }
        
        // 更新完成时间预测
        if (completionData.success && completionData.prediction) {
            const prediction = completionData.prediction;
            document.getElementById('estimatedCompletionTime').textContent = `${prediction.predicted_hours || 0}小时`;
            document.getElementById('completionTimeDesc').textContent = `约${prediction.predicted_weeks || 0}周 (每周${weeklyHours}小时)`;
            
            // 更新完成时间图表
            updateCompletionTimeChart(prediction);
        }
        
        // 获取每个未完成模块的掌握概率
        if (currentLearningPath && currentLearningPath.modules) {
            const completedModules = currentLearningPath.completed_modules || [];
            const modulePromises = currentLearningPath.modules
                .filter(module => !completedModules.includes(module.id))
                .map(module => 
                    fetch(API_ENDPOINTS.LEARNING_PATH_PREDICT_MASTERY, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            module_id: module.id,
                            weekly_hours: weeklyHours,
                            focus_level: 'medium'
                        })
                    })
                    .then(res => res.json())
                    .then(data => {
                        if (data.success && data.prediction) {
                            return { 
                                module_id: module.id, 
                                probability: data.prediction.probability || 0
                            };
                        }
                        return { module_id: module.id, probability: 0 };
                    })
                    .catch(err => {
                        console.error(`获取模块 ${module.id} 掌握概率失败:`, err);
                        return { module_id: module.id, probability: 0 };
                    })
                );
            
            // 等待所有模块预测完成
            const modulePredictions = await Promise.all(modulePromises);
            
            // 更新模块掌握概率显示
            modulePredictions.forEach(prediction => {
                const moduleElement = document.querySelector(`.start-module-btn[data-module-id="${prediction.module_id}"]`);
                if (moduleElement) {
                    const probabilityElement = moduleElement.parentElement.querySelector('.tooltip span');
                    if (probabilityElement) {
                        probabilityElement.innerHTML = `
                            <i class="fas fa-graduation-cap text-primary-hex"></i>
                            掌握概率: ${Math.round(prediction.probability * 100)}%
                        `;
                    }
                }
            });
            
            // 更新整体掌握概率
            const avgProbability = modulePredictions.reduce((sum, p) => sum + p.probability, 0) / 
                                  (modulePredictions.length || 1);
            document.getElementById('masteryProbability').textContent = `${Math.round(avgProbability * 100)}%`;
            document.getElementById('masteryDesc').textContent = `基于每周${weeklyHours}小时学习强度`;
            
            // 更新掌握概率图表
            updateMasteryProbabilityChart(modulePredictions, currentLearningPath.modules);
        }
    } catch (error) {
        console.error('更新学习路径预测失败:', error);
        showToast('警告', `更新预测数据失败: ${error.message}`, 'warning');
        
        // 恢复默认显示
        document.getElementById('masteryProbability').textContent = '计算中...';
    }
}

/**
 * 更新完成时间图表
 * @param {Object} prediction 完成时间预测数据
 */
function updateCompletionTimeChart(prediction) {
    // 检查图表容器是否存在
    const chartContainer = document.getElementById('completionTimeChart');
    if (!chartContainer) {
        console.warn('完成时间图表容器不存在');
        return;
    }
    
    // 销毁现有图表
    if (learningPathCharts.completionTime) {
        learningPathCharts.completionTime.destroy();
    }
    
    // 准备图表数据
    const weeklyHours = [5, 10, 15, 20];
    const predictedWeeks = weeklyHours.map(hours => {
        // 简单估计：如果每周x小时需要y周，那么每周z小时需要(x*y/z)周
        const baseHours = prediction.weekly_study_hours || 10;
        const baseWeeks = prediction.predicted_weeks || 0;
        return baseHours * baseWeeks / hours;
    });
    
    // 创建图表
    const ctx = chartContainer.getContext('2d');
    learningPathCharts.completionTime = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: weeklyHours.map(h => `每周${h}小时`),
            datasets: [{
                label: '预计完成周数',
                data: predictedWeeks,
                backgroundColor: 'rgba(2, 132, 199, 0.7)',
                borderColor: 'rgba(2, 132, 199, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '预计周数'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const weeks = context.raw;
                            const days = Math.round((weeks % 1) * 7);
                            return `预计完成时间: ${Math.floor(weeks)}周${days > 0 ? ` ${days}天` : ''}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * 更新掌握概率图表
 * @param {Array} predictions 模块掌握概率预测数据
 * @param {Array} modules 模块列表
 */
function updateMasteryProbabilityChart(predictions, modules) {
    // 检查图表容器是否存在
    const chartContainer = document.getElementById('masteryProbabilityChart');
    if (!chartContainer) {
        console.warn('掌握概率图表容器不存在');
        return;
    }
    
    // 销毁现有图表
    if (learningPathCharts.masteryProbability) {
        learningPathCharts.masteryProbability.destroy();
    }
    
    // 准备图表数据
    const moduleMap = new Map(modules.map(m => [m.id, m]));
    const chartData = predictions.map(p => {
        const module = moduleMap.get(p.module_id) || {};
        return {
            name: module.name || p.module_id,
            probability: Math.round(p.probability * 100)
        };
    });
    
    // 按概率排序
    chartData.sort((a, b) => b.probability - a.probability);
    
    // 限制显示数量
    const displayData = chartData.slice(0, 5);
    
    // 创建图表
    const ctx = chartContainer.getContext('2d');
    learningPathCharts.masteryProbability = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: displayData.map(d => d.name),
            datasets: [{
                label: '掌握概率',
                data: displayData.map(d => d.probability),
                backgroundColor: 'rgba(16, 185, 129, 0.7)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '掌握概率 (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function initQuerySubmission() {
    // 绑定提交按钮点击事件
    DOM.submitQueryButton().addEventListener('click', handleQuerySubmit);
    
    // 绑定输入框Enter键事件
    DOM.queryInput().addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuerySubmit();
        }
    });
    
    // 绑定查询模式变更事件
    DOM.queryModeSelector().forEach(radio => {
        radio.addEventListener('change', () => {
            updateQueryInputState();
        });
    });
    
    // 初始化查询输入状态
    updateQueryInputState();
}

/**
 * 处理查询提交
 * @param {Event} e 事件对象
 */
async function handleQuerySubmit(e) {
    if (e && e.preventDefault) {
    e.preventDefault();
    }
    
    // 获取查询输入
    const queryInput = DOM.queryInput();
    const query = queryInput.value.trim();
    
    // 验证输入
    if (!query) {
        showToast('错误', '请输入查询内容', 'error');
        return;
    }
    
    // 获取查询模式
    const queryMode = document.querySelector('input[name="queryMode"]:checked').value;
    
    // 构建请求数据
    const requestData = {
        query: query,
        mode: queryMode
    };
    
    // 如果是数据分析模式，添加数据和模型信息
    if (queryMode === 'data_analysis') {
        // 情况1: 只有数据，没有模型和目标列 - 数据处理建议
        if (currentData.path && !selectedModelName && !selectedTargetColumn) {
            requestData.data_analysis_type = 'data_consultation';
            requestData.data_path = currentData.path;
            requestData.data_preview = currentData.preview?.slice(0, 5); // 发送前5行样本
            requestData.columns = currentData.columns;
            requestData.column_types = currentData.columnTypes;
            
            console.log("📊 数据咨询模式 - 发送数据样本到大模型");
            
        // 情况2: 有数据和模型(可能还有目标列) - 生成教程
        } else if (currentData.path && selectedModelName) {
            requestData.data_analysis_type = 'tutorial_generation';
        requestData.data_path = currentData.path;
        requestData.model_name = selectedModelName;
            requestData.data_preview = currentData.preview?.slice(0, 5);
            requestData.columns = currentData.columns;
        
        if (selectedTargetColumn) {
            requestData.target_column = selectedTargetColumn;
            }
            
            console.log("🎓 教程生成模式 - 生成数据分析教程");
            
        // 情况3: 完整配置 - 标准数据分析
        } else if (currentData.path && selectedModelName && selectedTargetColumn) {
            requestData.data_path = currentData.path;
            requestData.model_name = selectedModelName;
            requestData.target_column = selectedTargetColumn;
            
            console.log("🔬 标准分析模式 - 执行完整数据分析");
            
        // 情况4: 配置不完整 - 提示用户
        } else {
            requestData.needs_configuration = true;
        }
    }
    
    // 显示加载状态
    showLoadingState();
    
    // 禁用提交按钮
    const submitButton = DOM.submitQueryButton();
    const originalButtonHtml = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 处理中...';
    
    try {
        // 发送请求
        const response = await fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        // 解析响应
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '查询失败');
        }
        
        // 显示响应
        displayQueryResponse(data);
        
        // 如果创建了学习路径，显示特殊提示和响应
        if (data.path_created) {
            // 在对话标签页也显示响应
            const dialogueResponseContainer = DOM.queryResponseContainer();
            if (dialogueResponseContainer) {
                dialogueResponseContainer.classList.remove('hidden');
                dialogueResponseContainer.innerHTML = `
                    <div class="content-card p-6 animate__animated animate__fadeInUp">
                        <h3 class="text-lg font-medium text-label flex items-center mb-4">
                            <i class="fas fa-check-circle text-success mr-2"></i>
                            学习路径创建成功
                        </h3>
                        <div class="prose prose-sm max-w-none">
                            ${marked ? marked.parse(data.answer) : data.answer.replace(/\n/g, '<br>')}
                        </div>
                        <div class="mt-4 flex gap-3">
                            <button onclick="document.getElementById('tab-link-learningPath').click()" class="btn btn-primary btn-sm">
                                <i class="fas fa-route mr-2"></i>查看学习路径
                            </button>
                            <button onclick="document.getElementById('tab-link-results').click()" class="btn btn-outline btn-sm">
                                <i class="fas fa-chart-bar mr-2"></i>查看详细结果
                            </button>
                        </div>
                    </div>
                `;
            }
            
            showToast('成功', '学习路径已创建！请查看"我的路径"标签页', 'success', 8000);
            // 2秒后自动刷新学习路径标签页
            setTimeout(() => {
                if (typeof loadUserLearningPaths === 'function') {
                    loadUserLearningPaths();
                }
            }, 2000);
        } else {
        showToast('成功', '查询完成', 'success');
            // 切换到结果标签页
            document.getElementById('tab-link-results').click();
        }
        
    } catch (error) {
        console.error('查询失败:', error);
        
        // 显示错误消息
        displayErrorResponse(error.message);
        
        // 显示通知
        showToast('错误', `查询失败: ${error.message}`, 'error');
    } finally {
        // 恢复提交按钮
        submitButton.disabled = false;
        submitButton.innerHTML = originalButtonHtml;
        
        // 隐藏加载状态
        hideLoadingState();
    }
}

/**
 * 显示加载状态
 */
function showLoadingState() {
    const loadingContainer = DOM.loadingSpinnerContainer();
    const responseSection = DOM.responseSection();
    
    if (loadingContainer) {
        loadingContainer.classList.remove('hidden');
    }
    
    if (responseSection) {
        responseSection.classList.add('hidden');
    }
    }
    
/**
 * 隐藏加载状态
 */
function hideLoadingState() {
    const loadingContainer = DOM.loadingSpinnerContainer();
    
    if (loadingContainer) {
        loadingContainer.classList.add('hidden');
    }
}

/**
 * 显示查询响应
 * @param {Object} data 响应数据
 */
function displayQueryResponse(data) {
    hideLoadingState();
    
    const responseSection = DOM.responseSection();
    const responseText = DOM.responseText();
    
    if (!responseSection || !responseText) {
        console.error('响应显示区域不存在');
        return;
    }
    
    // 显示响应区域
    responseSection.classList.remove('hidden');
    
    // 处理响应文本
    if (data.answer) {
        // 使用marked库解析Markdown
        if (typeof marked !== 'undefined') {
            responseText.innerHTML = marked.parse(data.answer);
        } else {
            // 简单处理换行
            responseText.innerHTML = `<p>${data.answer.replace(/\n/g, '<br>')}</p>`;
        }
    } else {
        responseText.innerHTML = '<p class="text-muted">暂无分析结果。</p>';
    }
    
    // 处理源文档（RAG相关）
    const sourceArea = DOM.sourceDocumentsArea();
    const sourceList = DOM.sourceDocumentsList();
    const sourceMessage = DOM.sourceDocumentsMessage();
    
    if (data.source_documents && data.source_documents.length > 0) {
        if (sourceArea) sourceArea.classList.remove('hidden');
        if (sourceList) {
            sourceList.innerHTML = '';
            data.source_documents.forEach((doc, index) => {
                const docElement = document.createElement('div');
                docElement.className = 'bg-base-100 border border-base-300 rounded-lg p-4 mb-3';
                docElement.innerHTML = `
                    <h5 class="font-medium mb-2">文档 ${index + 1}</h5>
                    <p class="text-sm text-muted mb-2">${doc.content || '无内容'}</p>
                    <div class="flex items-center gap-2 text-xs text-muted">
                        ${doc.source ? `<span>来源: ${doc.source}</span>` : ''}
                        ${doc.score ? `<span>相关性: ${Math.round(doc.score * 100)}%</span>` : ''}
                    </div>
            `;
                sourceList.appendChild(docElement);
        });
        }
        if (sourceMessage) sourceMessage.classList.add('hidden');
    } else {
        if (sourceArea) sourceArea.classList.add('hidden');
        if (sourceMessage) {
            sourceMessage.classList.remove('hidden');
            sourceMessage.textContent = '当前查询未引用特定来源文档。';
        }
    }
    
    // 处理可视化（如果有）
    const vizArea = DOM.visualizationDisplayArea();
    if (data.charts || data.feature_importance || data.prediction) {
        if (vizArea) vizArea.classList.remove('hidden');
        // 可以在这里添加图表渲染逻辑
    } else {
        if (vizArea) vizArea.classList.add('hidden');
    }
    
    // 处理特殊响应类型
    if (data.needs_data_and_model) {
        // 显示数据和模型需求提示
        responseText.innerHTML += `
            <div class="alert alert-info mt-4">
                <div class="flex items-center">
                    <i class="fas fa-info-circle mr-2"></i>
                    <div>
                        <h3 class="font-bold">需要上传数据和选择模型</h3>
                        <div class="text-sm">请先点击"上传/选择数据"按钮上传您的数据文件，然后选择合适的机器学习模型。</div>
                    </div>
                </div>
            </div>
        `;
    }
    }
    
/**
 * 显示错误响应
 * @param {string} errorMessage 错误消息
 */
function displayErrorResponse(errorMessage) {
    hideLoadingState();
    
    const responseSection = DOM.responseSection();
    const responseText = DOM.responseText();
    
    if (!responseSection || !responseText) {
        return;
    }
    
    // 显示响应区域
    responseSection.classList.remove('hidden');
    
    // 显示错误信息
    responseText.innerHTML = `
        <div class="alert alert-error shadow-lg">
            <div>
                <i class="fas fa-exclamation-circle"></i>
                <span>查询失败: ${errorMessage}</span>
                </div>
            </div>
        `;
        
    // 隐藏其他区域
    const sourceArea = DOM.sourceDocumentsArea();
    const vizArea = DOM.visualizationDisplayArea();
    
    if (sourceArea) sourceArea.classList.add('hidden');
    if (vizArea) vizArea.classList.add('hidden');
}

/**
 * 渲染图表
 * @param {string} containerId 容器ID
 * @param {Object} chartData 图表数据
 */
function renderChart(containerId, chartData) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    let chart;
    
    switch (chartData.type) {
        case 'bar':
            chart = echarts.init(container);
            chart.setOption({
                title: {
                    text: chartData.title,
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'shadow'
                    }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: chartData.categories || [],
                    axisTick: {
                        alignWithLabel: true
                    }
                },
                yAxis: {
                    type: 'value'
                },
                series: [{
                    name: chartData.series_name || '',
                    type: 'bar',
                    data: chartData.data || []
                }]
            });
            break;
            
        case 'line':
            chart = echarts.init(container);
            chart.setOption({
                title: {
                    text: chartData.title,
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis'
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: chartData.categories || []
                },
                yAxis: {
                    type: 'value'
                },
                series: [{
                    name: chartData.series_name || '',
                    type: 'line',
                    data: chartData.data || [],
                    smooth: true
                }]
            });
            break;
            
        case 'pie':
            chart = echarts.init(container);
            chart.setOption({
                title: {
                    text: chartData.title,
                    left: 'center'
                },
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b}: {c} ({d}%)'
                },
                legend: {
                    orient: 'horizontal',
                    bottom: 0,
                    data: chartData.categories || []
                },
                series: [{
                    name: chartData.series_name || '',
                    type: 'pie',
                    radius: ['50%', '70%'],
                    avoidLabelOverlap: false,
                    label: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            fontSize: '12',
                            fontWeight: 'bold'
                        }
                    },
                    labelLine: {
                        show: false
                    },
                    data: (chartData.categories || []).map((cat, index) => ({
                        value: chartData.data[index],
                        name: cat
                    }))
                }]
            });
            break;
            
        case 'scatter':
            chart = echarts.init(container);
            chart.setOption({
                title: {
                    text: chartData.title,
                    left: 'center'
                },
                tooltip: {
                    trigger: 'item',
                    formatter: function(params) {
                        return params.seriesName + '<br/>' + 
                            (chartData.labels ? chartData.labels[params.dataIndex] + ': ' : '') +
                            params.value[0] + ', ' + params.value[1];
                    }
                },
                grid: {
                    left: '3%',
                    right: '7%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    name: chartData.x_label || '',
                    nameLocation: 'center',
                    nameGap: 30
                },
                yAxis: {
                    type: 'value',
                    name: chartData.y_label || '',
                    nameLocation: 'center',
                    nameGap: 30
                },
                series: [{
                    name: chartData.series_name || '',
                    type: 'scatter',
                    symbolSize: 10,
                    data: chartData.data || []
                }]
            });
            break;
            
        default:
            container.innerHTML = '<div class="alert alert-warning">不支持的图表类型</div>';
    }
    
    // 响应式调整
    if (chart) {
        window.addEventListener('resize', () => {
            chart.resize();
        });
    }
}

/**
 * 保存学习路径
 * @param {Object} learningPath 学习路径数据
 */
function saveLearningPath(learningPath) {
    // 构建请求数据
    const requestData = {
        path: learningPath
    };
    
    // 发送请求
    fetch(API_ENDPOINTS.SAVE_LEARNING_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('成功', '学习路径已保存', 'success');
            
            // 可选：刷新我的路径列表
            if (typeof loadMyLearningPaths === 'function') {
                loadMyLearningPaths();
            }
        } else {
            throw new Error(data.error || '保存失败');
        }
    })
    .catch(error => {
        console.error('保存学习路径失败:', error);
        showToast('错误', `保存学习路径失败: ${error.message}`, 'error');
    });
}

/**
 * HTML转义
 * @param {string} text 文本
 * @returns {string} 转义后的文本
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    
    return text.replace(/[&<>"']/g, m => map[m]);
}

function initModelSelectionDelegation() {
    // 使用事件委托来处理模型卡片点击事件
    DOM.modelGrid().addEventListener('click', (e) => {
        // 查找最近的模型卡片父元素
        const modelCard = e.target.closest('.model-card');
        if (!modelCard) return;
        
        // 获取模型名称
        const modelName = modelCard.getAttribute('data-model-name');
        if (!modelName) return;
        
        // 更新选中状态
        selectModel(modelName, modelCard);
    });
}

/**
 * 选择模型
 * @param {string} modelName 模型名称
 * @param {HTMLElement} modelCard 模型卡片元素
 */
function selectModel(modelName, modelCard) {
    // 验证参数
    if (!modelName || !modelCard) {
        console.error('selectModel: 缺少必要参数');
        return;
    }
    
    // 移除之前选中的模型
    const previousSelectedCards = document.querySelectorAll('.model-card.selected-model-card');
    previousSelectedCards.forEach(card => {
        card.classList.remove('selected-model-card');
        // 确保卡片恢复到正面
        const inner = card.querySelector('.model-card-inner');
        if (inner) {
            inner.classList.remove('no-flip');
        }
    });
    
    // 标记当前选中的模型
    modelCard.classList.add('selected-model-card');
    
    // 确保选中的卡片显示正面且不翻转
    const inner = modelCard.querySelector('.model-card-inner');
    if (inner) {
        inner.classList.add('no-flip');
        inner.style.transform = 'rotateY(0deg)';
    }
    
    // 更新全局状态
    selectedModelName = modelName;
    
    // 更新选中模型信息
    updateSelectedModelInfo(modelName);
    
    // 更新查询输入状态
    updateQueryInputState();
    
    // 显示通知
    const displayName = modelCard.getAttribute('data-display-name') || modelName;
    showToast('模型已选择', `已选择 ${displayName} 模型用于分析`, 'success');
}

/**
 * 更新选中模型信息
 * @param {string} modelName 模型名称
 */
function updateSelectedModelInfo(modelName) {
    const infoElement = DOM.selectedModelInfo();
    
    if (!infoElement) return;
    
    // 查找模型卡片以获取详细信息
    const modelCard = document.querySelector(`.model-card[data-model-name="${modelName}"]`);
    
    if (modelCard) {
        const displayName = modelCard.getAttribute('data-display-name') || modelName;
        const modelType = modelCard.getAttribute('data-model-type') || '';
        
        infoElement.innerHTML = `
            <div class="flex items-center justify-center gap-2">
                <i class="fas fa-check-circle text-success"></i>
                <span>当前已选择: <strong class="text-primary-hex">${displayName}</strong></span>
                ${modelType ? `<span class="badge badge-sm badge-primary">${modelType}</span>` : ''}
            </div>
        `;
    } else {
        infoElement.textContent = `当前已选择模型: ${modelName}`;
    }
}

/**
 * 更新查询输入状态
 */
function updateQueryInputState() {
    const queryInput = DOM.queryInput();
    const queryInputLabel = DOM.queryInputLabel();
    const submitQueryButton = DOM.submitQueryButton();
    const uploadDataShortcutBtn = DOM.uploadDataShortcutBtn();
    
    // 获取查询模式
    const queryMode = document.querySelector('input[name="queryMode"]:checked').value;
    
    // 如果是数据分析模式
    if (queryMode === 'data_analysis') {
        // 显示数据上传按钮
        if (uploadDataShortcutBtn) {
            uploadDataShortcutBtn.style.display = 'inline-flex';
        }
        
        // 检查是否已上传数据和选择模型
        const hasData = currentData.path !== null;
        const hasModel = selectedModelName !== null;
        
        if (!hasData && !hasModel) {
            // 没有数据和模型
            queryInput.placeholder = '请先上传数据并选择模型...';
            queryInputLabel.textContent = '您想解决什么问题？(需要先上传数据并选择模型)';
            submitQueryButton.disabled = true;
            
            DOM.modeSpecificInfo().innerHTML = `
                <div class="flex items-center gap-2 text-warning">
                    <i class="fas fa-info-circle"></i>
                    <span><strong>数据分析模式</strong> - 请先上传数据文件并选择分析模型</span>
                </div>
            `;
        } else if (!hasData) {
            // 有模型但没有数据
            queryInput.placeholder = '请先上传数据...';
            queryInputLabel.textContent = '您想解决什么问题？(需要先上传数据)';
            submitQueryButton.disabled = true;
            
            DOM.modeSpecificInfo().innerHTML = `
                <div class="flex items-center gap-2 text-warning">
                    <i class="fas fa-info-circle"></i>
                    <span>已选择模型: <strong>${selectedModelName}</strong>，还需要上传数据</span>
                </div>
            `;
        } else if (!hasModel) {
            // 有数据但没有模型
            queryInput.placeholder = '请先选择模型...';
            queryInputLabel.textContent = '您想解决什么问题？(需要先选择模型)';
            submitQueryButton.disabled = true;
            
            DOM.modeSpecificInfo().innerHTML = `
                <div class="flex items-center gap-2 text-warning">
                    <i class="fas fa-info-circle"></i>
                    <span>已上传数据: <strong>${currentData.fileName || currentData.path}</strong>，还需要选择分析模型</span>
                </div>
            `;
        } else {
            // 数据和模型都已准备就绪
            queryInput.placeholder = '请输入您的问题，例如："如何使用这个模型分析我的数据？"或"帮我分析数据的特征分布"';
            queryInputLabel.textContent = '您想解决什么问题？';
            submitQueryButton.disabled = false;
            
            // 更新模式特定信息
            DOM.modeSpecificInfo().innerHTML = `
                <div class="flex flex-col gap-2">
                    <div class="flex items-center gap-2 text-success">
                        <i class="fas fa-check-circle"></i>
                        <span><strong>数据分析模式</strong> - 系统已就绪</span>
                    </div>
                <div class="flex items-center gap-2">
                        <i class="fas fa-database text-primary-hex"></i>
                        <span>数据文件: <strong>${currentData.fileName || currentData.path}</strong></span>
                </div>
                    <div class="flex items-center gap-2">
                    <i class="fas fa-robot text-primary-hex"></i>
                        <span>分析模型: <strong>${selectedModelName}</strong></span>
                </div>
                ${selectedTargetColumn ? `
                    <div class="flex items-center gap-2">
                    <i class="fas fa-bullseye text-primary-hex"></i>
                    <span>目标列: <strong>${selectedTargetColumn}</strong></span>
                </div>
                ` : ''}
                </div>
            `;
        }
    } else {
        // 通用大模型问答模式
        // 隐藏数据上传按钮
        if (uploadDataShortcutBtn) {
            uploadDataShortcutBtn.style.display = 'none';
        }
        
        queryInput.placeholder = '请输入您的问题，我将尽力回答...';
        queryInputLabel.textContent = '您想了解什么？';
        submitQueryButton.disabled = false;
        
        // 更新模式特定信息
        DOM.modeSpecificInfo().innerHTML = `
            <div class="flex items-center gap-2 text-info">
                <i class="fas fa-brain text-secondary-hex"></i>
                <span><strong>通用大模型问答</strong> - 可以询问任何问题，无需上传数据或选择模型</span>
            </div>
            <div class="mt-2 text-xs text-muted">
                <span>💡 适合知识问答、概念解释、学习指导等通用查询</span>
            </div>
        `;
    }
}

/**
 * 初始化文件上传表单
 */
function initUploadForm() {
    const uploadForm = DOM.uploadForm();
    
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        const fileInput = DOM.dataFile();
        
        // 检查是否选择了文件
        if (!fileInput.files || fileInput.files.length === 0) {
            showToast('错误', '请选择文件', 'error');
            return;
        }
        
        // 获取文件对象
        const file = fileInput.files[0];
        
        // 检查文件类型
        const allowedTypes = ['.csv', '.xlsx', '.xls', '.json'];
        const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!allowedTypes.includes(fileExt)) {
            showToast('错误', `不支持的文件类型: ${fileExt}，仅支持 CSV, Excel 和 JSON`, 'error');
            return;
        }
        
        // 显示加载状态
        const analyzeBtn = DOM.analyzeDataBtn();
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 上传中...';
        
        try {
            // 上传文件
            const response = await fetch(API_ENDPOINTS.UPLOAD, {
                method: 'POST',
                body: formData
            });
            
            // 解析响应
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || '上传失败');
            }
            
            // 更新数据状态
            currentData = {
                path: data.file_path,
                fileName: data.original_filename,
                columns: data.columns || [],
                columnTypes: data.column_types || {},
                categorical_columns: data.categorical_columns || [],
                numerical_columns: data.numerical_columns || [],
                preview: data.preview || [],
                rowCount: data.row_count || 0,
                columnCount: data.column_count || 0,
                analysisCompleted: false
            };
            
            // 显示数据预览
            displayDataPreview(currentData.preview, currentData.columns);
            
            // 更新查询输入状态
            updateQueryInputState();
            
            // 显示通知
            showToast('上传成功', '数据文件上传成功', 'success');
            
            // 分析数据
            analyzeData(currentData.path);
        } catch (error) {
            console.error('上传文件失败:', error);
            showToast('错误', `上传文件失败: ${error.message}`, 'error');
        } finally {
            // 恢复按钮状态
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-cogs"></i>分析数据';
        }
    });
}

/**
 * 显示数据预览
 * @param {Array} previewData 预览数据
 * @param {Array} columns 列名列表
 */
function displayDataPreview(previewData, columns) {
    const previewElement = DOM.dataPreview();
    
    if (!previewElement) return;
    
    // 如果没有数据或列，显示提示信息
    if (!previewData || previewData.length === 0 || !columns || columns.length === 0) {
        previewElement.innerHTML = '<p class="text-muted p-4 text-center">无数据可预览。</p>';
        return;
    }
    
    // 创建表格
    let tableHtml = '<div class="overflow-x-auto"><table class="table table-compact w-full">';
    
    // 表头
    tableHtml += '<thead><tr>';
    columns.forEach(col => {
        tableHtml += `<th>${col}</th>`;
    });
    tableHtml += '</tr></thead>';
    
    // 表体
    tableHtml += '<tbody>';
    previewData.forEach(row => {
        tableHtml += '<tr>';
        columns.forEach(col => {
            // 安全处理可能的空值
            const value = row[col] !== undefined && row[col] !== null ? row[col] : '';
            tableHtml += `<td>${value}</td>`;
        });
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table></div>';
    
    // 更新预览元素
    previewElement.innerHTML = tableHtml;
}

/**
 * 分析数据
 * @param {string} filePath 文件路径
 */
async function analyzeData(filePath) {
    try {
        // 显示加载状态
        const analyzeBtn = DOM.analyzeDataBtn();
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 分析中...';
        
        // 调用API分析数据
        const response = await fetch(API_ENDPOINTS.ANALYZE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_path: filePath })
        });
        
        // 解析响应
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '分析失败');
        }
        
        // 更新数据分析结果
        currentData.analysisCompleted = true;
        
        // 显示分析结果
        displayAnalysisResults(data);
        
        // 显示通知
        showToast('分析完成', '数据分析完成', 'success');
    } catch (error) {
        console.error('分析数据失败:', error);
        showToast('错误', `分析数据失败: ${error.message}`, 'error');
    } finally {
        // 恢复按钮状态
        const analyzeBtn = DOM.analyzeDataBtn();
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-cogs"></i>分析数据';
    }
}

/**
 * 显示数据分析结果
 * @param {Object} data 分析结果数据
 */
function displayAnalysisResults(data) {
    // 显示结果区域
    DOM.dataAnalysisResults().classList.remove('hidden');
    
    // 更新行数和列数
    DOM.rowCount().textContent = data.row_count || 0;
    DOM.columnCount().textContent = data.column_count || 0;
    
    // 更新推荐模型
    if (data.recommended_models && data.recommended_models.length > 0) {
        DOM.recommendedModels().innerHTML = data.recommended_models
            .map(model => `<span class="badge badge-primary">${model}</span>`)
            .join(' ');
    } else {
        DOM.recommendedModels().innerHTML = '<span class="text-muted">无推荐模型</span>';
    }
    
    // 更新目标列选择器
    if (data.columns && data.columns.length > 0) {
        const targetSelector = DOM.targetColumnSelector();
        targetSelector.innerHTML = '';
        
        // 添加每个列作为可能的目标
        data.columns.forEach(col => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-sm';
            btn.textContent = col;
            btn.setAttribute('data-column', col);
            
            // 绑定点击事件
            btn.addEventListener('click', () => {
                // 移除之前选中的目标列
                targetSelector.querySelectorAll('.btn-active').forEach(b => {
                    b.classList.remove('btn-active');
                });
                
                // 标记当前选中的目标列
                btn.classList.add('btn-active');
                
                // 更新全局状态
                selectedTargetColumn = col;
                
                // 更新查询输入状态
                updateQueryInputState();
                
                // 显示通知
                showToast('目标列已选择', `已选择 ${col} 作为目标列`, 'success');
            });
            
            targetSelector.appendChild(btn);
        });
    }
}

/**
 * 初始化示例查询
 */
function initExampleQueries() {
    const exampleQueryList = DOM.exampleQueryList();
    
    if (!exampleQueryList) return;
    
    // 定义示例查询
    const examples = [
        {
            text: "解释这个数据集的主要特征",
            category: "数据分析"
        },
        {
            text: "使用这个模型预测目标列的最佳方法是什么？",
            category: "模型应用"
        },
        {
            text: "哪些特征对预测结果影响最大？",
            category: "特征分析"
        },
        {
            text: "如何提高模型的准确率？",
            category: "模型优化"
        },
        {
            text: "我想学习机器学习，我目前没有相关背景，每周可以学习10小时左右，帮我制定一个学习路径。",
            category: "学习规划"
        },
        {
            text: "解释线性回归和逻辑回归的区别",
            category: "机器学习概念"
        },
        {
            text: "如何处理数据中的缺失值？",
            category: "数据预处理"
        },
        {
            text: "特征工程的最佳实践有哪些？",
            category: "特征工程"
        }
    ];
    
    // 按类别分组
    const groupedExamples = examples.reduce((acc, example) => {
        if (!acc[example.category]) {
            acc[example.category] = [];
        }
        acc[example.category].push(example);
        return acc;
    }, {});
    
    // 清空列表
    exampleQueryList.innerHTML = '';
    
    // 添加每个类别的示例查询
    Object.entries(groupedExamples).forEach(([category, categoryExamples]) => {
        // 添加类别标题
        const categoryItem = document.createElement('li');
        categoryItem.className = 'menu-title';
        categoryItem.innerHTML = `<span>${category}</span>`;
        exampleQueryList.appendChild(categoryItem);
        
        // 添加示例查询
        categoryExamples.forEach(example => {
            const item = document.createElement('li');
            const link = document.createElement('a');
            link.textContent = example.text;
            
            // 绑定点击事件
            link.addEventListener('click', () => {
                // 将示例查询填入输入框
                DOM.queryInput().value = example.text;
                DOM.queryInput().focus();
            });
            
            item.appendChild(link);
            exampleQueryList.appendChild(item);
        });
    });
}

/**
 * 创建模型卡片元素
 * @param {string} internalName 内部名称
 * @param {string} displayName 显示名称
 * @param {string} modelType 模型类型
 * @param {string} description 描述
 * @param {string} iconClass 图标类
 * @returns {string} 模型卡片HTML
 */
function createModelCardElement(internalName, displayName, modelType, description, iconClass) {
    return `
        <div class="model-card" data-model-name="${internalName}" data-display-name="${displayName}" data-model-type="${modelType}">
            <div class="model-card-inner">
                <div class="model-card-front">
                    <div class="model-icon">
                        <i class="fas ${iconClass || 'fa-cube'}"></i>
                    </div>
                    <h4>${displayName || internalName}</h4>
                    <p class="text-gray-500 text-xs mt-1">${modelType || ''}</p>
                </div>
                <div class="model-card-back">
                    <h4>${displayName || internalName}</h4>
                    <p>${description || '没有描述'}</p>
                </div>
            </div>
        </div>
    `;
}

/**
 * 获取模型类别显示名称
 * @param {string} category 类别键名
 * @returns {string} 显示名称
 */
function getCategoryDisplayName(category) {
    const displayNames = {
        "regression": "回归模型",
        "classification": "分类模型",
        "clustering": "聚类模型",
        "ensemble": "集成模型",
        "other": "其他模型"
    };
    
    return displayNames[category] || category;
}

/**
 * 获取模型显示名称
 * @param {string} internalName 内部名称
 * @returns {string} 显示名称
 */
function getModelDisplayName(internalName) {
    const displayNames = {
        "linear_regression": "线性回归",
        "logistic_regression": "逻辑回归",
        "knn_classifier": "K近邻分类器",
        "decision_tree": "决策树",
        "svm_classifier": "支持向量机",
        "naive_bayes": "朴素贝叶斯",
        "random_forest_classifier": "随机森林分类器",
        "random_forest_regressor": "随机森林回归器",
        "kmeans": "K-Means聚类"
    };
    
    return displayNames[internalName] || internalName;
}

/**
 * 获取默认模型图标
 * @param {string} internalName 内部名称
 * @returns {string} 图标类名
 */
function getDefaultModelIcon(internalName) {
    const iconMap = {
        "linear_regression": "fa-chart-line",
        "logistic_regression": "fa-code-branch",
        "knn_classifier": "fa-project-diagram",
        "decision_tree": "fa-sitemap",
        "random_forest_classifier": "fa-tree",
        "random_forest_regressor": "fa-tree",
        "svm_classifier": "fa-vector-square",
        "naive_bayes": "fa-percentage",
        "kmeans": "fa-object-group"
    };
    
    return iconMap[internalName] || "fa-cube";
}

/**
 * 显示toast通知
 * @param {string} title 标题
 * @param {string} message 消息内容
 * @param {string} type 类型 (success, error, warning, info)
 * @param {number} duration 显示时长 (毫秒)
 */
function showToast(title, message, type = 'info', duration = 5000) {
    // 获取toast容器
    const container = DOM.toastContainer();
    
    if (!container) return;
    
    // 生成唯一ID
    const toastId = `toast-${Date.now()}`;
    
    // 创建toast元素
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `alert shadow-lg ${getAlertClass(type)} mb-3 animate__animated animate__fadeInRight ring-2 ring-sky-300/30`;
    toast.innerHTML = `
        <div>
            <i class="${getAlertIcon(type)}"></i>
            <div>
                <h3 class="font-bold">${title}</h3>
                <div class="text-xs">${message}</div>
            </div>
        </div>
        <button class="btn btn-sm btn-circle btn-ghost transition-transform hover:scale-110" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // 添加到容器
    container.appendChild(toast);
    
    // 设置自动关闭
    if (toastTimeouts[toastId]) {
        clearTimeout(toastTimeouts[toastId]);
    }
    
    toastTimeouts[toastId] = setTimeout(() => {
        // 添加淡出动画
        toast.classList.remove('animate__fadeInRight');
        toast.classList.add('animate__fadeOutRight');
        
        // 移除元素
        setTimeout(() => {
            if (toast.parentElement) {
                toast.parentElement.removeChild(toast);
            }
            delete toastTimeouts[toastId];
        }, 500);
    }, duration);
}

/**
 * 获取警告类名
 * @param {string} type 类型
 * @returns {string} 类名
 */
function getAlertClass(type) {
    switch (type) {
        case 'success': return 'alert-success';
        case 'error': return 'alert-error';
        case 'warning': return 'alert-warning';
        default: return 'alert-info';
    }
}

/**
 * 获取警告图标
 * @param {string} type 类型
 * @returns {string} 图标类名
 */
function getAlertIcon(type) {
    switch (type) {
        case 'success': return 'fas fa-check-circle';
        case 'error': return 'fas fa-exclamation-circle';
        case 'warning': return 'fas fa-exclamation-triangle';
        default: return 'fas fa-info-circle';
    }
}

// 添加到文档加载完成后执行main函数
document.addEventListener('DOMContentLoaded', main);

/**
 * 初始化技术实验室功能
 */
function initTechLabFeatures() {
    // 学习场景选择器事件监听
    const scenarioSelect = document.getElementById('learningScenarioSelect');
    const customParams = document.getElementById('customScenarioParams');
    
    if (scenarioSelect && customParams) {
        scenarioSelect.addEventListener('change', () => {
            if (scenarioSelect.value === 'custom') {
                customParams.classList.remove('hidden');
            } else {
                customParams.classList.add('hidden');
            }
        });
    }
    
    // 集成策略选择器事件监听
    const ensembleRadios = document.querySelectorAll('input[name="ensembleStrategy"]');
    const stackingOptions = document.getElementById('stackingOptionsContainer');
    
    ensembleRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === 'stacking' && radio.checked) {
                if (stackingOptions) stackingOptions.classList.remove('hidden');
            } else {
                if (stackingOptions) stackingOptions.classList.add('hidden');
            }
        });
    });
    
    // 添加模型按钮事件监听
    const addModelBtn = document.getElementById('addModelBtn');
    const baseModelsContainer = document.getElementById('baseModelsContainer');
    
    if (addModelBtn && baseModelsContainer) {
        addModelBtn.addEventListener('click', () => {
            const newModelItem = document.createElement('div');
            newModelItem.className = 'model-select-item flex items-center gap-3 border rounded-lg p-3';
            newModelItem.innerHTML = `
                <div class="form-control grow">
                    <select class="select select-bordered select-sm w-full base-model-select">
                        <option value="" disabled selected>选择模型...</option>
                        <option value="linear_regression">线性回归</option>
                        <option value="logistic_regression">逻辑回归</option>
                        <option value="decision_tree">决策树</option>
                        <option value="random_forest">随机森林</option>
                        <option value="svm">支持向量机</option>
                        <option value="knn">K近邻</option>
                    </select>
                </div>
                <button class="btn btn-xs btn-circle btn-ghost remove-model-btn">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            // 添加删除按钮事件
            const removeBtn = newModelItem.querySelector('.remove-model-btn');
            removeBtn.addEventListener('click', () => {
                newModelItem.remove();
            });
            
            baseModelsContainer.appendChild(newModelItem);
        });
    }
    
    // 为现有的删除按钮添加事件监听
    document.querySelectorAll('.remove-model-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const modelItems = document.querySelectorAll('.model-select-item');
            if (modelItems.length > 1) {
                btn.closest('.model-select-item').remove();
            } else {
                showToast('提示', '至少需要保留一个模型', 'info');
            }
        });
    });
    
    // 运行模拟按钮事件监听 - 已移至AI辅助版本
    // const runSimulationBtn = document.getElementById('runSimulationBtn');
    // if (runSimulationBtn) {
    //     runSimulationBtn.addEventListener('click', runSimulation);
    // }
    
    // 重置模拟按钮事件监听
    const resetSimulationBtn = document.getElementById('resetSimulationBtn');
    if (resetSimulationBtn) {
        resetSimulationBtn.addEventListener('click', resetSimulation);
    }
    
    // 导出数据按钮事件监听
    const exportBtn = document.getElementById('exportSimulationDataBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportSimulationData);
    }
    
    // 保存实验按钮事件监听
    const saveBtn = document.getElementById('saveSimulationBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveSimulation);
    }
    
    // 模型文档查看按钮事件监听
    document.querySelectorAll('.view-model-docs-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const modelName = btn.getAttribute('data-model');
            showModelDocumentation(modelName);
        });
    });
    
    // 模型详情模态框关闭按钮
    const closeModalBtn = document.getElementById('closeModelDetailBtn');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            const modal = document.getElementById('modelDetailModal');
            if (modal) modal.classList.remove('modal-open');
        });
    }
    
    // 文档标签页切换
    document.querySelectorAll('#modelDocsContent .tabs .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // 移除所有活动状态
            document.querySelectorAll('#modelDocsContent .tabs .tab').forEach(t => t.classList.remove('tab-active'));
            // 添加当前活动状态
            tab.classList.add('tab-active');
            
            // 获取标签页ID
            const tabId = tab.id;
            switchDocsTab(tabId);
        });
    });
    
    // 运行模拟按钮添加AI辅助
    const runSimulationBtn = document.getElementById('runSimulationBtn');
    if (runSimulationBtn) {
        runSimulationBtn.addEventListener('click', runSimulationWithAI);
        runSimulationBtn.innerHTML = '<i class="fas fa-robot mr-2"></i>AI辅助模拟';
        runSimulationBtn.classList.add('btn-shimmer', 'pulse-glow');
    }
}

/**
 * 切换文档标签页
 */
function switchDocsTab(tabId) {
    // 隐藏所有文档内容
    document.querySelectorAll('.docs-tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // 显示对应内容
    let targetContentId = '';
    switch(tabId) {
        case 'tab-docs-models':
            targetContentId = 'docs-content-models';
            break;
        case 'tab-docs-ensemble':
            targetContentId = 'docs-content-ensemble';
            break;
        case 'tab-docs-features':
            targetContentId = 'docs-content-features';
            break;
    }
    
    const targetContent = document.getElementById(targetContentId);
    if (targetContent) {
        targetContent.classList.remove('hidden');
    }
}

/**
 * 使用AI辅助运行模拟实验
 */
async function runSimulationWithAI() {
    try {
        const runBtn = document.getElementById('runSimulationBtn');
        const originalHtml = runBtn.innerHTML;
        
        // 显示加载状态
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> AI分析中...';
        
        // 收集完整的实验参数
        const params = collectDetailedSimulationParams();
        
        if (!params.baseModels || params.baseModels.length < 1) {
            showToast('错误', '请至少选择一个基础模型', 'error');
            return;
        }
        
        // 构建详细的AI分析请求
        const analysisRequest = {
            query: `请根据我的机器学习实验配置进行深度分析，包括参数优化建议、模型选择评价、实验结果预测等`,
            mode: 'general_llm',
            experiment_context: {
                prediction_target: params.predictionTarget,
                scenario: params.scenario,
                ensemble_strategy: params.ensembleStrategy,
                base_models: params.baseModels,
                custom_params: params.customParams,
                experiment_type: 'machine_learning_simulation',
                detailed_config: {
                    dataset_type: params.datasetType || 'synthetic',
                    evaluation_metrics: params.evaluationMetrics || [],
                    cross_validation: params.crossValidation || false,
                    feature_selection: params.featureSelection || false,
                    hyperparameter_tuning: params.hyperparameterTuning || false
                }
            },
            analysis_requirements: [
                '分析模型组合的合理性',
                '评估集成策略的适用性', 
                '预测可能的性能表现',
                '提供参数调优建议',
                '指出潜在的问题和风险',
                '推荐最佳实践方法'
            ]
        };
        
        // 第一步：请求AI进行实验分析
        console.log('📊 发送AI分析请求...', analysisRequest);
        
        const analysisResponse = await fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analysisRequest)
        });
        
        const analysisData = await analysisResponse.json();
        
        if (!analysisResponse.ok) {
            throw new Error(analysisData.error || 'AI分析请求失败');
        }
        
        // 第二步：基于AI建议运行增强模拟
        runBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 执行模拟实验...';
        
        // 模拟实验运行过程
        await simulateExperimentProcess(params);
        
        // 第三步：生成AI增强的结果
        const results = generateAIEnhancedResults(params, analysisData.answer);
        
        // 第四步：请求AI进行结果解读
        const interpretationRequest = {
            query: '请深度解读以下机器学习实验结果，包括性能分析、模型比较、改进建议等',
            mode: 'general_llm',
            experiment_results: {
                model_performance: results.modelResults,
                experiment_params: params,
                best_model: results.bestModel,
                performance_summary: results.performanceSummary
            }
        };
        
        const interpretationResponse = await fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(interpretationRequest)
        });
        
        const interpretationData = await interpretationResponse.json();
        
        if (interpretationResponse.ok && interpretationData.answer) {
            results.aiInterpretation = interpretationData.answer;
        }
        
        // 将AI分析结果融入到实验结果中
        results.aiAnalysis = analysisData.answer;
        results.aiEnhanced = true;
        
        // 显示增强结果
        displayAIEnhancedResults(results);
        
        showToast('AI分析完成', 'AI辅助实验分析已完成，查看详细结果', 'success', 6000);
        
    } catch (error) {
        console.error('AI辅助模拟失败:', error);
        showToast('错误', `AI辅助模拟失败: ${error.message}`, 'error');
        
        // 回退到普通模拟
        console.log('回退到普通模拟...');
        await runSimulation();
    } finally {
        // 恢复按钮状态
        const runBtn = document.getElementById('runSimulationBtn');
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fas fa-robot mr-2"></i>AI辅助模拟';
    }
}

/**
 * 运行模拟实验
 */
async function runSimulation() {
    try {
        const runBtn = document.getElementById('runSimulationBtn');
        const originalHtml = runBtn.innerHTML;
        
        // 显示加载状态
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> 运行中...';
        
        // 收集实验参数
        const params = collectSimulationParams();
        
        if (!params.baseModels || params.baseModels.length < 1) {
            showToast('错误', '请至少选择一个基础模型', 'error');
            return;
        }
        
        // 模拟运行过程（这里使用模拟数据，实际应用中应调用后端API）
        await new Promise(resolve => setTimeout(resolve, 2000)); // 模拟2秒处理时间
        
        // 生成模拟结果
        const results = generateSimulationResults(params);
        
        // 显示结果
        displaySimulationResults(results);
        
        showToast('成功', '模拟实验完成', 'success');
        
    } catch (error) {
        console.error('运行模拟失败:', error);
        showToast('错误', `运行模拟失败: ${error.message}`, 'error');
    } finally {
        // 恢复按钮状态
        const runBtn = document.getElementById('runSimulationBtn');
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fas fa-play-circle mr-2"></i>运行模拟';
    }
}

/**
 * 收集模拟参数
 */
function collectSimulationParams() {
    const predictionTarget = document.getElementById('predictionTargetSelect').value;
    const scenario = document.getElementById('learningScenarioSelect').value;
    const ensembleStrategy = document.querySelector('input[name="ensembleStrategy"]:checked').value;
    
    // 收集选中的基础模型
    const baseModels = [];
    document.querySelectorAll('.base-model-select').forEach(select => {
        if (select.value) {
            baseModels.push(select.value);
        }
    });
    
    // 收集自定义参数（如果是自定义场景）
    let customParams = {};
    if (scenario === 'custom') {
        customParams = {
            difficulty: document.getElementById('contentDifficultySelect').value,
            priorKnowledge: document.getElementById('priorKnowledgeSelect').value,
            weeklyHours: parseInt(document.getElementById('weeklyHoursInput').value),
            focusLevel: document.getElementById('focusLevelSelect').value
        };
    }
    
    return {
        predictionTarget,
        scenario,
        ensembleStrategy,
        baseModels,
        customParams
    };
}

/**
 * 收集详细的模拟参数（AI增强版本）
 */
function collectDetailedSimulationParams() {
    const basicParams = collectSimulationParams();
    
    // 扩展参数收集
    const detailedParams = {
        ...basicParams,
        
        // 数据集配置
        datasetType: 'synthetic', // 可以从UI获取
        sampleSize: 1000, // 可以从UI获取
        featureCount: 10, // 可以从UI获取
        
        // 评估配置
        evaluationMetrics: basicParams.predictionTarget === 'mastery_probability' 
            ? ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            : ['mse', 'mae', 'r2_score', 'rmse'],
        
        // 实验配置
        crossValidation: true,
        cvFolds: 5,
        testSize: 0.2,
        randomState: 42,
        
        // 高级选项
        featureSelection: false,
        hyperparameterTuning: false,
        featureScaling: true,
        handleMissingValues: true,
        
        // 集成学习特定参数
        ensembleConfig: getEnsembleConfig(basicParams.ensembleStrategy),
        
        // 时间戳和实验ID
        experimentId: `exp_${Date.now()}`,
        timestamp: new Date().toISOString()
    };
    
    return detailedParams;
}

/**
 * 获取集成学习配置
 */
function getEnsembleConfig(strategy) {
    const configs = {
        'voting': {
            voting: 'hard', // 或 'soft' 对于概率投票
            weights: null // 等权重
        },
        'averaging': {
            method: 'mean', // 或 'median'
            weights: null
        },
        'stacking': {
            meta_learner: 'logistic_regression',
            use_probas: true,
            cv_folds: 3
        }
    };
    
    return configs[strategy] || configs['voting'];
}

/**
 * 模拟实验运行过程
 */
async function simulateExperimentProcess(params) {
    const steps = [
        '数据预处理',
        '特征工程',
        '模型训练',
        '交叉验证',
        '集成学习',
        '性能评估'
    ];
    
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        const progress = ((i + 1) / steps.length) * 100;
        
        // 更新UI显示当前步骤
        const runBtn = document.getElementById('runSimulationBtn');
        runBtn.innerHTML = `<span class="loading loading-spinner loading-xs"></span> ${step}... (${Math.round(progress)}%)`;
        
        // 模拟处理时间
        await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));
    }
}

/**
 * 生成AI增强的实验结果
 */
function generateAIEnhancedResults(params, aiAnalysis) {
    const basicResults = generateSimulationResults(params);
    
    // 增强结果数据
    const enhancedResults = {
        ...basicResults,
        
        // AI分析建议
        aiAnalysis: aiAnalysis,
        
        // 更详细的性能指标
        detailedMetrics: generateDetailedMetrics(basicResults),
        
        // 模型复杂度分析
        complexityAnalysis: generateComplexityAnalysis(params.baseModels),
        
        // 训练时间模拟
        trainingTimes: generateTrainingTimes(params.baseModels),
        
        // 特征重要性（模拟）
        featureImportance: generateFeatureImportance(),
        
        // 学习曲线数据
        learningCurves: generateLearningCurves(params.baseModels),
        
        // 性能摘要
        performanceSummary: generatePerformanceSummary(basicResults),
        
        // 最佳模型
        bestModel: identifyBestModel(basicResults),
        
        // 实验配置记录
        experimentConfig: params,
        
        // AI增强标记
        aiEnhanced: true,
        enhancedAt: new Date().toISOString()
    };
    
    return enhancedResults;
}

/**
 * 生成详细性能指标
 */
function generateDetailedMetrics(basicResults) {
    return basicResults.modelResults.map(result => {
        if (basicResults.isClassification) {
            return {
                ...result,
                auc_roc: (Math.random() * 0.3 + 0.7).toFixed(3),
                specificity: (Math.random() * 0.2 + 0.8).toFixed(3),
                npv: (Math.random() * 0.2 + 0.8).toFixed(3),
                balanced_accuracy: (Math.random() * 0.2 + 0.8).toFixed(3)
            };
        } else {
            return {
                ...result,
                mape: (Math.random() * 15 + 5).toFixed(2),
                explained_variance: (Math.random() * 0.2 + 0.8).toFixed(3),
                max_error: (Math.random() * 20 + 10).toFixed(2),
                median_ae: (Math.random() * 5 + 2).toFixed(3)
            };
        }
    });
}

/**
 * 生成模型复杂度分析
 */
function generateComplexityAnalysis(models) {
    const complexityMap = {
        'linear_regression': { complexity: 'Low', parameters: 'Few', interpretability: 'High' },
        'logistic_regression': { complexity: 'Low', parameters: 'Few', interpretability: 'High' },
        'decision_tree': { complexity: 'Medium', parameters: 'Medium', interpretability: 'High' },
        'random_forest': { complexity: 'High', parameters: 'Many', interpretability: 'Medium' },
        'svm': { complexity: 'High', parameters: 'Medium', interpretability: 'Low' },
        'knn': { complexity: 'Medium', parameters: 'Few', interpretability: 'Medium' }
    };
    
    return models.map(model => ({
        model: getModelDisplayName(model),
        ...complexityMap[model] || { complexity: 'Medium', parameters: 'Medium', interpretability: 'Medium' }
    }));
}

/**
 * 生成训练时间模拟
 */
function generateTrainingTimes(models) {
    const baseTimeMap = {
        'linear_regression': 0.1,
        'logistic_regression': 0.2,
        'decision_tree': 0.5,
        'random_forest': 2.0,
        'svm': 1.5,
        'knn': 0.3
    };
    
    return models.map(model => ({
        model: getModelDisplayName(model),
        training_time: ((baseTimeMap[model] || 1.0) * (0.8 + Math.random() * 0.4)).toFixed(2)
    }));
}

/**
 * 生成特征重要性（模拟）
 */
function generateFeatureImportance() {
    const features = ['学习时间', '先验知识', '专注度', '难度等级', '练习频率', '概念理解', '记忆能力', '问题解决'];
    
    return features.map(feature => ({
        feature: feature,
        importance: Math.random().toFixed(3),
        rank: 0
    })).sort((a, b) => b.importance - a.importance).map((item, index) => ({
        ...item,
        rank: index + 1
    }));
}

/**
 * 生成学习曲线数据
 */
function generateLearningCurves(models) {
    const trainingSizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    return models.map(model => {
        const baseScore = 0.6 + Math.random() * 0.3;
        const trainScores = trainingSizes.map(size => 
            Math.min(0.99, baseScore + size * 0.2 + (Math.random() - 0.5) * 0.1)
        );
        const testScores = trainingSizes.map((size, i) => 
            Math.min(trainScores[i] - 0.05, baseScore + size * 0.15 + (Math.random() - 0.5) * 0.1)
        );
        
        return {
            model: getModelDisplayName(model),
            training_sizes: trainingSizes,
            train_scores: trainScores.map(s => s.toFixed(3)),
            test_scores: testScores.map(s => s.toFixed(3))
        };
    });
}

/**
 * 生成性能摘要
 */
function generatePerformanceSummary(results) {
    const metricKey = results.isClassification ? 'accuracy' : 'r2';
    const scores = results.modelResults.map(r => parseFloat(r[metricKey]));
    
    return {
        best_score: Math.max(...scores).toFixed(3),
        worst_score: Math.min(...scores).toFixed(3),
        average_score: (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3),
        score_std: Math.sqrt(scores.reduce((sq, n) => sq + Math.pow(n - (scores.reduce((a, b) => a + b, 0) / scores.length), 2), 0) / scores.length).toFixed(3),
        model_count: results.modelResults.length
    };
}

/**
 * 识别最佳模型
 */
function identifyBestModel(results) {
    const metricKey = results.isClassification ? 'accuracy' : 'r2';
    let bestModel = results.modelResults[0];
    let bestScore = parseFloat(bestModel[metricKey]);
    
    results.modelResults.forEach(model => {
        const score = parseFloat(model[metricKey]);
        if (score > bestScore) {
            bestScore = score;
            bestModel = model;
        }
    });
    
    return {
        name: bestModel.model,
        score: bestScore,
        metric: metricKey,
        all_metrics: bestModel
    };
}

/**
 * 生成模拟结果（模拟数据）
 */
function generateSimulationResults(params) {
    const models = params.baseModels;
    const isClassification = params.predictionTarget === 'mastery_probability';
    
    // 生成模拟的性能指标
    const modelResults = models.map(model => {
        const baseScore = Math.random() * 0.3 + 0.6; // 60%-90%的基础分数
        
        if (isClassification) {
            return {
                model: getModelDisplayName(model),
                accuracy: (baseScore + Math.random() * 0.1).toFixed(3),
                precision: (baseScore + Math.random() * 0.1).toFixed(3),
                recall: (baseScore + Math.random() * 0.1).toFixed(3),
                f1: (baseScore + Math.random() * 0.1).toFixed(3)
            };
        } else {
            return {
                model: getModelDisplayName(model),
                mse: (Math.random() * 10 + 5).toFixed(3),
                r2: baseScore.toFixed(3),
                mae: (Math.random() * 5 + 2).toFixed(3),
                rmse: (Math.random() * 8 + 3).toFixed(3)
            };
        }
    });
    
    // 集成模型结果（通常比单个模型好一些）
    const ensembleResult = (() => {
        if (isClassification) {
            const avgAccuracy = modelResults.reduce((sum, r) => sum + parseFloat(r.accuracy), 0) / modelResults.length;
            return {
                model: `集成模型 (${params.ensembleStrategy})`,
                accuracy: Math.min(0.99, avgAccuracy + 0.05).toFixed(3),
                precision: Math.min(0.99, avgAccuracy + 0.03).toFixed(3),
                recall: Math.min(0.99, avgAccuracy + 0.04).toFixed(3),
                f1: Math.min(0.99, avgAccuracy + 0.04).toFixed(3)
            };
        } else {
            const avgMse = modelResults.reduce((sum, r) => sum + parseFloat(r.mse), 0) / modelResults.length;
            const avgR2 = modelResults.reduce((sum, r) => sum + parseFloat(r.r2), 0) / modelResults.length;
            return {
                model: `集成模型 (${params.ensembleStrategy})`,
                mse: Math.max(1, avgMse - 2).toFixed(3),
                r2: Math.min(0.99, avgR2 + 0.05).toFixed(3),
                mae: Math.max(0.5, avgMse * 0.6).toFixed(3),
                rmse: Math.max(1, avgMse * 0.8).toFixed(3)
            };
        }
    })();
    
    modelResults.push(ensembleResult);
    
    return {
        modelResults,
        isClassification,
        params
    };
}

/**
 * 显示模拟结果
 */
function displaySimulationResults(results) {
    const resultsSection = document.getElementById('simulationResultsSection');
    const metricsTableBody = document.getElementById('metricsTableBody');
    const analysisText = document.getElementById('simulationAnalysisText');
    
    // 显示结果区域
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
    }
    
    // 更新性能指标表格
    if (metricsTableBody) {
        metricsTableBody.innerHTML = '';
        
        results.modelResults.forEach(result => {
            const row = document.createElement('tr');
            if (results.isClassification) {
                row.innerHTML = `
                    <td class="font-medium">${result.model}</td>
                    <td>${result.accuracy}</td>
                    <td>${result.precision}</td>
                    <td>${result.recall}</td>
                `;
            } else {
                row.innerHTML = `
                    <td class="font-medium">${result.model}</td>
                    <td>${result.mse}</td>
                    <td>${result.r2}</td>
                    <td>${result.mae}</td>
                `;
            }
            metricsTableBody.appendChild(row);
        });
    }
    
    // 更新结果分析
    if (analysisText) {
        const bestModel = results.modelResults[results.modelResults.length - 1]; // 集成模型通常是最后一个
        const taskType = results.isClassification ? '分类' : '回归';
        const metricName = results.isClassification ? '准确率' : 'R²得分';
        const bestScore = results.isClassification ? bestModel.accuracy : bestModel.r2;
        
        let analysisContent = `
            <h4 class="font-medium mb-3">实验结果分析</h4>
            <p class="mb-3">
                本次实验比较了 ${results.modelResults.length - 1} 个基础模型在${taskType}任务上的表现，
                并使用${results.params.ensembleStrategy}策略构建了集成模型。
            </p>
            <p class="mb-3">
                <strong>最佳模型：</strong> ${bestModel.model}，${metricName}达到 ${bestScore}
            </p>
        `;
        
        // 如果有AI分析结果，添加到显示中
        if (results.aiAnalysis) {
            analysisContent += `
                <div class="border-t pt-4 mt-4">
                    <h5 class="font-medium mb-3 text-primary-hex">🤖 AI专家分析</h5>
                    <div class="bg-base-200 p-4 rounded-lg prose prose-sm max-w-none">
                        ${marked ? marked.parse(results.aiAnalysis) : results.aiAnalysis.replace(/\n/g, '<br>')}
                    </div>
                </div>
            `;
        }
        
        analysisContent += `
            <p class="mb-3">
                <strong>关键发现：</strong>
            </p>
            <ul class="list-disc list-inside mb-3 space-y-1">
                <li>集成模型通过结合多个基础模型的预测，通常能获得比单个模型更好的性能</li>
                <li>${results.params.ensembleStrategy}策略在此场景下表现良好</li>
                <li>不同模型在相同数据上的表现存在差异，体现了模型选择的重要性</li>
            </ul>
            <p class="text-sm text-muted">
                <strong>注意：</strong> 以上结果基于模拟数据生成，实际应用中需要使用真实数据进行验证。
            </p>
        `;
        
        analysisText.innerHTML = analysisContent;
    }
    
    // 更新图表
    updateSimulationCharts(results);
}

/**
 * 更新模拟图表
 */
function updateSimulationCharts(results) {
    if (!results || !results.modelResults) {
        console.warn('updateSimulationCharts: 缺少图表数据');
        return;
    }
    
    const chartCanvas = document.getElementById('modelComparisonChart');
    if (!chartCanvas) {
        console.warn('模型对比图表容器不存在');
        return;
    }
    
    // 检查Chart.js是否已加载
    if (typeof Chart === 'undefined') {
        console.error('Chart.js库未加载，无法创建图表');
        chartCanvas.parentElement.innerHTML = `
            <div class="alert alert-warning">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle mr-2"></i>
                    <span>图表库未加载，请刷新页面重试</span>
                </div>
            </div>
        `;
        return;
    }
    
    try {
        const ctx = chartCanvas.getContext('2d');
        if (!ctx) {
            throw new Error('无法获取canvas上下文');
        }
        
        // 销毁现有图表
        if (window.simulationChart) {
            window.simulationChart.destroy();
            window.simulationChart = null;
        }
        
        // 准备图表数据
        const labels = results.modelResults.map(r => r.model || '未知模型');
        const metricKey = results.isClassification ? 'accuracy' : 'r2';
        const data = results.modelResults.map(r => {
            const value = parseFloat(r[metricKey]);
            return isNaN(value) ? 0 : value;
        });
        
        // 验证数据
        if (labels.length === 0 || data.length === 0) {
            throw new Error('图表数据为空');
        }
        
        // 创建新图表
        window.simulationChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: results.isClassification ? '准确率' : 'R²得分',
                    data: data,
                    backgroundColor: 'rgba(2, 132, 199, 0.7)',
                    borderColor: 'rgba(2, 132, 199, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: results.isClassification ? 1 : Math.max(...data.filter(d => !isNaN(d))) * 1.1,
                        title: {
                            display: true,
                            text: results.isClassification ? '准确率' : 'R²得分'
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                if (isNaN(value)) return '数据无效';
                                
                                const percentage = results.isClassification ? 
                                    `${(value * 100).toFixed(1)}%` : 
                                    value.toFixed(3);
                                return `${context.dataset.label}: ${percentage}`;
                            }
                        }
                    }
                }
            }
        });
        
        // 添加窗口大小调整监听器
        const resizeHandler = () => {
            if (window.simulationChart) {
                window.simulationChart.resize();
            }
        };
        
        // 移除旧的监听器（如果存在）
        if (window.chartResizeHandler) {
            window.removeEventListener('resize', window.chartResizeHandler);
        }
        
        // 添加新的监听器
        window.chartResizeHandler = resizeHandler;
        window.addEventListener('resize', resizeHandler);
        
    } catch (error) {
        console.error('创建图表失败:', error);
        // 显示友好的错误信息
        chartCanvas.parentElement.innerHTML = `
            <div class="alert alert-error">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle mr-2"></i>
                    <span>图表渲染失败: ${error.message}</span>
                </div>
                <div class="mt-2">
                    <button onclick="location.reload()" class="btn btn-sm btn-outline">刷新页面</button>
                </div>
            </div>
        `;
    }
}

/**
 * 重置模拟
 */
function resetSimulation() {
    const resultsSection = document.getElementById('simulationResultsSection');
    if (resultsSection) {
        resultsSection.classList.add('hidden');
    }
    
    // 销毁图表
    if (window.simulationChart) {
        window.simulationChart.destroy();
        window.simulationChart = null;
    }
    
    showToast('提示', '模拟已重置', 'info');
}

/**
 * 导出模拟数据
 */
function exportSimulationData() {
    showToast('提示', '导出功能正在开发中', 'info');
}

/**
 * 保存模拟实验
 */
function saveSimulation() {
    showToast('提示', '保存功能正在开发中', 'info');
}

/**
 * 显示模型文档
 */
function showModelDocumentation(modelName) {
    const modal = document.getElementById('modelDetailModal');
    const title = document.getElementById('modelDetailTitle');
    const content = document.getElementById('modelDetailContent');
    
    if (!modal || !title || !content) return;
    
    // 模型文档数据
    const modelDocs = {
        linear_regression: {
            title: '线性回归模型',
            content: `
                <h4 class="font-medium mb-2">模型原理</h4>
                <p class="mb-3">线性回归通过找到最佳拟合直线来预测连续数值，使用最小二乘法优化参数。</p>
                
                <h4 class="font-medium mb-2">适用场景</h4>
                <ul class="list-disc list-inside mb-3">
                    <li>学习时间预测</li>
                    <li>特征与目标变量呈线性关系</li>
                    <li>需要模型可解释性的场景</li>
                </ul>
                
                <h4 class="font-medium mb-2">优缺点</h4>
                <p class="mb-2"><strong>优点：</strong>简单易懂、计算快速、可解释性强</p>
                <p><strong>缺点：</strong>假设线性关系、对异常值敏感</p>
            `
        },
        logistic_regression: {
            title: '逻辑回归模型',
            content: `
                <h4 class="font-medium mb-2">模型原理</h4>
                <p class="mb-3">逻辑回归使用逻辑函数将线性组合映射到概率值，适用于二分类和多分类问题。</p>
                
                <h4 class="font-medium mb-2">适用场景</h4>
                <ul class="list-disc list-inside mb-3">
                    <li>掌握概率预测</li>
                    <li>通过/不通过分类</li>
                    <li>需要概率输出的场景</li>
                </ul>
                
                <h4 class="font-medium mb-2">优缺点</h4>
                <p class="mb-2"><strong>优点：</strong>输出概率、不假设数据分布、训练快速</p>
                <p><strong>缺点：</strong>假设线性决策边界、对特征工程要求高</p>
            `
        },
        decision_tree: {
            title: '决策树模型',
            content: `
                <h4 class="font-medium mb-2">模型原理</h4>
                <p class="mb-3">决策树通过递归分割特征空间，构建树状决策结构来进行预测。</p>
                
                <h4 class="font-medium mb-2">适用场景</h4>
                <ul class="list-disc list-inside mb-3">
                    <li>非线性关系捕捉</li>
                    <li>分类和回归任务</li>
                    <li>需要决策规则解释的场景</li>
                </ul>
                
                <h4 class="font-medium mb-2">优缺点</h4>
                <p class="mb-2"><strong>优点：</strong>可解释性强、处理非线性关系、无需特征缩放</p>
                <p><strong>缺点：</strong>容易过拟合、对噪声敏感</p>
            `
        },
        random_forest: {
            title: '随机森林模型',
            content: `
                <h4 class="font-medium mb-2">模型原理</h4>
                <p class="mb-3">随机森林通过构建多个决策树并投票/平均来提高预测精度和泛化能力。</p>
                
                <h4 class="font-medium mb-2">适用场景</h4>
                <ul class="list-disc list-inside mb-3">
                    <li>复杂的学习预测任务</li>
                    <li>需要高精度的场景</li>
                    <li>特征重要性分析</li>
                </ul>
                
                <h4 class="font-medium mb-2">优缺点</h4>
                <p class="mb-2"><strong>优点：</strong>精度高、抗过拟合、提供特征重要性</p>
                <p><strong>缺点：</strong>模型复杂、训练时间长、可解释性较差</p>
            `
        }
    };
    
    const doc = modelDocs[modelName];
    if (doc) {
        title.textContent = doc.title;
        content.innerHTML = doc.content;
        modal.classList.add('modal-open');
    }
}

/**
 * 更新文档内容
 */
function updateDocsContent(tabId) {
    // 这里可以根据不同的标签页显示不同的内容
    // 当前保持现有内容不变
}

/**
 * 显示AI增强的模拟结果 - 优化版本
 */
function displayAIEnhancedResults(results) {
    const resultsSection = document.getElementById('simulationResultsSection');
    const metricsTableBody = document.getElementById('metricsTableBody');
    const analysisText = document.getElementById('simulationAnalysisText');
    
    // 显示结果区域
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
    }
    
    // 更新增强版交互式表格
    if (metricsTableBody) {
        // 清空现有内容
        metricsTableBody.innerHTML = '';
        
        // 创建表格容器包装器
        const tableContainer = metricsTableBody.closest('.overflow-x-auto') || metricsTableBody.closest('div');
        if (tableContainer) {
            tableContainer.innerHTML = `
                <div class="mb-4 flex flex-wrap gap-3 items-center justify-between">
                    <div class="flex gap-2 items-center">
                        <span class="text-sm font-medium">排序:</span>
                        <select id="tableSortSelect" class="select select-bordered select-sm">
                            <option value="default">默认</option>
                            <option value="performance-desc">性能降序</option>
                            <option value="performance-asc">性能升序</option>
                            <option value="name">模型名称</option>
                        </select>
                    </div>
                    <div class="flex gap-2 items-center">
                        <span class="text-sm font-medium">筛选:</span>
                        <select id="tableFilterSelect" class="select select-bordered select-sm">
                            <option value="all">全部模型</option>
                            <option value="ensemble">仅集成模型</option>
                            <option value="base">仅基础模型</option>
                            <option value="top3">性能前三</option>
                        </select>
                    </div>
                    <div class="flex gap-2">
                        <button id="exportTableBtn" class="btn btn-sm btn-outline">
                            <i class="fas fa-download mr-1"></i>导出
                        </button>
                        <button id="refreshTableBtn" class="btn btn-sm btn-primary">
                            <i class="fas fa-sync-alt mr-1"></i>刷新
                        </button>
                    </div>
                </div>
                
                <div class="card bg-base-100 shadow-lg border border-base-300">
                    <div class="card-body p-0">
                        <div class="overflow-x-auto">
                            <table class="table table-zebra w-full" id="enhancedResultsTable">
                                <thead class="bg-gradient-to-r from-primary/10 to-secondary/10">
                                    <tr>
                                        <th class="sortable cursor-pointer hover:bg-primary/20 transition-colors" data-sort="model">
                                            模型名称 
                                            <i class="fas fa-sort ml-1 text-xs opacity-60"></i>
                                        </th>
                                        <th class="sortable cursor-pointer hover:bg-primary/20 transition-colors" data-sort="accuracy">
                                            ${results.isClassification ? '准确率' : 'R²得分'}
                                            <i class="fas fa-sort ml-1 text-xs opacity-60"></i>
                                        </th>
                                        <th class="sortable cursor-pointer hover:bg-primary/20 transition-colors" data-sort="precision">
                                            ${results.isClassification ? '精确率' : 'MAE'}
                                            <i class="fas fa-sort ml-1 text-xs opacity-60"></i>
                                        </th>
                                        <th class="sortable cursor-pointer hover:bg-primary/20 transition-colors" data-sort="recall">
                                            ${results.isClassification ? '召回率' : 'RMSE'}
                                            <i class="fas fa-sort ml-1 text-xs opacity-60"></i>
                                        </th>
                                        <th class="sortable cursor-pointer hover:bg-primary/20 transition-colors" data-sort="f1">
                                            ${results.isClassification ? 'F1得分' : 'MSE'}
                                            <i class="fas fa-sort ml-1 text-xs opacity-60"></i>
                                        </th>
                                        <th>扩展指标</th>
                                        <th>训练时间</th>
                                        <th>复杂度</th>
                                        <th>操作</th>
                                    </tr>
                                </thead>
                                <tbody id="enhancedTableBody">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- 表格统计信息 -->
                <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="stat bg-gradient-to-r from-success/10 to-success/5 border border-success/20 rounded-lg">
                        <div class="stat-title text-success">最佳性能</div>
                        <div class="stat-value text-lg text-success" id="bestPerformanceValue">-</div>
                        <div class="stat-desc text-success/70" id="bestPerformanceModel">-</div>
                    </div>
                    <div class="stat bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 rounded-lg">
                        <div class="stat-title text-primary">平均性能</div>
                        <div class="stat-value text-lg text-primary" id="avgPerformanceValue">-</div>
                        <div class="stat-desc text-primary/70">所有模型平均</div>
                    </div>
                    <div class="stat bg-gradient-to-r from-warning/10 to-warning/5 border border-warning/20 rounded-lg">
                        <div class="stat-title text-warning">性能差异</div>
                        <div class="stat-value text-lg text-warning" id="performanceVarianceValue">-</div>
                        <div class="stat-desc text-warning/70">标准差</div>
                    </div>
                    <div class="stat bg-gradient-to-r from-info/10 to-info/5 border border-info/20 rounded-lg">
                        <div class="stat-title text-info">集成提升</div>
                        <div class="stat-value text-lg text-info" id="ensembleImprovementValue">-</div>
                        <div class="stat-desc text-info/70">相对基础模型</div>
                    </div>
                </div>
            `;
        }
        
        // 渲染表格数据
        renderEnhancedTableData(results);
        
        // 绑定表格交互事件
        bindTableInteractions(results);
        
        // 更新统计信息
        updateTableStatistics(results);
    }
    
    // 更新AI增强的结果分析（保持原有逻辑）
    if (analysisText) {
        const taskType = results.isClassification ? '分类' : '回归';
        const metricName = results.isClassification ? '准确率' : 'R²得分';
        
        let analysisContent = `
            <div class="space-y-6">
                <!-- 实验概述 -->
                <div class="card bg-gradient-to-r from-primary/10 to-secondary/10 shadow-lg border border-primary/20">
                    <div class="card-body">
                        <h4 class="text-lg font-bold mb-4 flex items-center">
                            <i class="fas fa-chart-line text-primary-hex mr-2"></i>
                            🤖 AI深度分析报告
                        </h4>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="stat bg-white/50 rounded-lg p-4">
                                <div class="stat-title text-xs">最佳性能</div>
                                <div class="stat-value text-lg text-success">${results.performanceSummary.best_score}</div>
                                <div class="stat-desc text-xs">模型: ${results.bestModel.name}</div>
                            </div>
                            <div class="stat bg-white/50 rounded-lg p-4">
                                <div class="stat-title text-xs">平均性能</div>
                                <div class="stat-value text-lg">${results.performanceSummary.average_score}</div>
                                <div class="stat-desc text-xs">标准差: ${results.performanceSummary.score_std}</div>
                            </div>
                            <div class="stat bg-white/50 rounded-lg p-4">
                                <div class="stat-title text-xs">模型数量</div>
                                <div class="stat-value text-lg text-info">${results.performanceSummary.model_count}</div>
                                <div class="stat-desc text-xs">包含集成模型</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- AI专家分析 -->
                <div class="card bg-gradient-to-r from-accent/10 to-warning/10 shadow-lg border border-accent/20">
                    <div class="card-body">
                        <h5 class="text-lg font-bold mb-4 flex items-center">
                            <i class="fas fa-robot text-accent mr-2"></i>
                            🎯 专业洞察与建议
                        </h5>
                        <div class="prose prose-sm max-w-none bg-white/30 p-4 rounded-lg">
                            ${marked ? marked.parse(results.aiAnalysis) : results.aiAnalysis.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                </div>
        `;
        
        // 如果有AI结果解读，添加到显示中
        if (results.aiInterpretation) {
            analysisContent += `
                <!-- AI结果解读 -->
                <div class="card bg-gradient-to-r from-success/10 to-info/10 shadow-lg border border-success/20">
                    <div class="card-body">
                        <h5 class="text-lg font-bold mb-4 flex items-center">
                            <i class="fas fa-microscope text-success mr-2"></i>
                            📊 实验结果深度解读
                        </h5>
                        <div class="prose prose-sm max-w-none bg-white/30 p-4 rounded-lg">
                            ${marked ? marked.parse(results.aiInterpretation) : results.aiInterpretation.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // 性能对比雷达图
        analysisContent += `
                <!-- 多维性能对比 -->
                <div class="card bg-base-100 shadow-lg border border-base-300">
                    <div class="card-body">
                        <h5 class="text-lg font-bold mb-4 flex items-center">
                            <i class="fas fa-radar-chart text-secondary mr-2"></i>
                            📈 多维性能分析
                        </h5>
                        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div>
                                <canvas id="performanceRadarChart" width="400" height="300"></canvas>
                            </div>
                            <div class="space-y-4">
                                <h6 class="font-medium">性能评估维度</h6>
                                <div class="space-y-3">
                                    ${results.detailedMetrics.map((model, index) => `
                                        <div class="flex items-center justify-between p-3 bg-base-200/50 rounded-lg">
                                            <span class="font-medium text-sm">${model.model}</span>
                                            <div class="flex gap-2">
                                                <span class="badge badge-sm ${getPerformanceColor(model, results.isClassification)}">
                                                    ${results.isClassification ? model.accuracy : model.r2}
                                                </span>
                                                <span class="text-xs text-muted">排名 ${index + 1}</span>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="text-sm text-muted bg-base-200/50 p-4 rounded-lg border-l-4 border-primary">
                    <strong>💡 实验说明：</strong> 本实验采用AI增强分析，结合了多维度性能评估和专业建议。
                    所有指标均基于模拟数据生成，在实际应用中建议使用真实数据进行验证。
                </div>
            </div>
        `;
        
        analysisText.innerHTML = analysisContent;
        
        // 创建雷达图
        setTimeout(() => createPerformanceRadarChart(results), 100);
    }
    
    // 更新AI增强图表
    updateAIEnhancedCharts(results);
    
    // 更新快速统计卡片
    updateQuickStatistics(results);
}

/**
 * 渲染增强表格数据
 */
function renderEnhancedTableData(results) {
    const tableBody = document.getElementById('enhancedTableBody');
    if (!tableBody || !results.detailedMetrics) return;
    
    tableBody.innerHTML = '';
    
    results.detailedMetrics.forEach((result, index) => {
        const isEnsemble = result.model.includes('集成模型');
        const trainingTime = results.trainingTimes?.find(t => t.model === result.model)?.training_time || '未知';
        const complexity = results.complexityAnalysis?.find(c => c.model === result.model)?.complexity || 'Medium';
        
        const row = document.createElement('tr');
        row.className = `hover:bg-base-100 transition-all duration-200 ${isEnsemble ? 'bg-primary/5 border-l-4 border-l-primary' : ''}`;
        row.setAttribute('data-model-type', isEnsemble ? 'ensemble' : 'base');
        
        if (results.isClassification) {
            row.innerHTML = `
                <td class="font-medium ${isEnsemble ? 'text-primary-hex' : ''}">
                    <div class="flex items-center gap-2">
                        ${isEnsemble ? '<i class="fas fa-layer-group text-primary-hex text-xs"></i>' : '<i class="fas fa-cog text-muted text-xs"></i>'}
                        <span>${result.model}</span>
                        ${isEnsemble ? '<span class="badge badge-primary badge-xs">集成</span>' : ''}
                    </div>
                </td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(result.accuracy)}">${result.accuracy}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(result.precision)}">${result.precision}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(result.recall)}">${result.recall}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(result.f1)}">${result.f1}</span></td>
                <td class="text-xs">
                    <div class="tooltip" data-tip="AUC-ROC: ${result.auc_roc || '-'}">
                        <span class="cursor-help">AUC: ${result.auc_roc || '-'}</span>
                    </div>
                </td>
                <td class="text-xs text-muted">${trainingTime}秒</td>
                <td><span class="badge badge-xs ${getComplexityBadgeClass(complexity)}">${complexity}</span></td>
                <td>
                    <div class="dropdown dropdown-end">
                        <button class="btn btn-xs btn-ghost" tabindex="0">
                            <i class="fas fa-ellipsis-v"></i>
                        </button>
                        <ul class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-32">
                            <li><a onclick="viewModelDetails('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-info-circle"></i>详情
                            </a></li>
                            <li><a onclick="compareModel('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-balance-scale"></i>对比
                            </a></li>
                            <li><a onclick="exportModelData('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-download"></i>导出
                            </a></li>
                        </ul>
                    </div>
                </td>
            `;
        } else {
            row.innerHTML = `
                <td class="font-medium ${isEnsemble ? 'text-primary-hex' : ''}">
                    <div class="flex items-center gap-2">
                        ${isEnsemble ? '<i class="fas fa-layer-group text-primary-hex text-xs"></i>' : '<i class="fas fa-cog text-muted text-xs"></i>'}
                        <span>${result.model}</span>
                        ${isEnsemble ? '<span class="badge badge-primary badge-xs">集成</span>' : ''}
                    </div>
                </td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(result.r2)}">${result.r2}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(1 - parseFloat(result.mae) / 50, true)}">${result.mae}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(1 - parseFloat(result.rmse) / 50, true)}">${result.rmse}</span></td>
                <td><span class="badge badge-sm ${getScoreBadgeClass(1 - parseFloat(result.mse) / 100, true)}">${result.mse}</span></td>
                <td class="text-xs">
                    <div class="tooltip" data-tip="MAPE: ${result.mape || '-'}%">
                        <span class="cursor-help">MAPE: ${result.mape || '-'}%</span>
                    </div>
                </td>
                <td class="text-xs text-muted">${trainingTime}秒</td>
                <td><span class="badge badge-xs ${getComplexityBadgeClass(complexity)}">${complexity}</span></td>
                <td>
                    <div class="dropdown dropdown-end">
                        <button class="btn btn-xs btn-ghost" tabindex="0">
                            <i class="fas fa-ellipsis-v"></i>
                        </button>
                        <ul class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-32">
                            <li><a onclick="viewModelDetails('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-info-circle"></i>详情
                            </a></li>
                            <li><a onclick="compareModel('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-balance-scale"></i>对比
                            </a></li>
                            <li><a onclick="exportModelData('${result.model.replace(/'/g, "\\'")}')">
                                <i class="fas fa-download"></i>导出
                            </a></li>
                        </ul>
                    </div>
                </td>
            `;
        }
        
        tableBody.appendChild(row);
    });
}

/**
 * 获取分数等级的徽章样式
 */
function getScoreBadgeClass(score, inverse = false) {
    const numScore = parseFloat(score);
    if (inverse) {
        // 对于错误率等指标，值越小越好
        if (numScore > 0.8) return 'badge-success';
        if (numScore > 0.6) return 'badge-warning';
        return 'badge-error';
    } else {
        // 对于准确率等指标，值越大越好
        if (numScore > 0.9) return 'badge-success';
        if (numScore > 0.8) return 'badge-warning';
        if (numScore > 0.7) return 'badge-info';
        return 'badge-error';
    }
}

/**
 * 获取复杂度等级的徽章样式
 */
function getComplexityBadgeClass(complexity) {
    switch (complexity.toLowerCase()) {
        case 'low': return 'badge-success';
        case 'medium': return 'badge-warning';
        case 'high': return 'badge-error';
        default: return 'badge-neutral';
    }
}

/**
 * 更新AI增强图表
 */
function updateAIEnhancedCharts(results) {
    // 复用现有的图表更新函数，但传入增强的结果
    updateSimulationCharts(results);
    
    // 可以在这里添加额外的AI增强图表
    // 例如：学习曲线图、特征重要性图等
}

/**
 * 绑定表格交互事件
 */
function bindTableInteractions(results) {
    // 排序功能
    const sortSelect = document.getElementById('tableSortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            sortTable(results, sortSelect.value);
        });
    }
    
    // 筛选功能
    const filterSelect = document.getElementById('tableFilterSelect');
    if (filterSelect) {
        filterSelect.addEventListener('change', () => {
            filterTable(results, filterSelect.value);
        });
    }
    
    // 导出功能
    const exportBtn = document.getElementById('exportTableBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            exportTableData(results);
        });
    }
    
    // 刷新功能
    const refreshBtn = document.getElementById('refreshTableBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            renderEnhancedTableData(results);
            updateTableStatistics(results);
            showToast('刷新完成', '表格数据已刷新', 'success', 2000);
        });
    }
    
    // 表头排序点击
    document.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const sortKey = th.getAttribute('data-sort');
            sortTableByColumn(results, sortKey);
        });
    });
}

/**
 * 更新表格统计信息
 */
function updateTableStatistics(results) {
    if (!results.performanceSummary) return;
    
    const bestValue = document.getElementById('bestPerformanceValue');
    const bestModel = document.getElementById('bestPerformanceModel');
    const avgValue = document.getElementById('avgPerformanceValue');
    const varianceValue = document.getElementById('performanceVarianceValue');
    const ensembleImprovement = document.getElementById('ensembleImprovementValue');
    
    if (bestValue) bestValue.textContent = results.performanceSummary.best_score;
    if (bestModel) bestModel.textContent = results.bestModel.name;
    if (avgValue) avgValue.textContent = results.performanceSummary.average_score;
    if (varianceValue) varianceValue.textContent = results.performanceSummary.score_std;
    
    // 计算集成模型相对基础模型的提升
    if (ensembleImprovement && results.detailedMetrics) {
        const ensembleModel = results.detailedMetrics.find(m => m.model.includes('集成模型'));
        const baseModels = results.detailedMetrics.filter(m => !m.model.includes('集成模型'));
        
        if (ensembleModel && baseModels.length > 0) {
            const metricKey = results.isClassification ? 'accuracy' : 'r2';
            const ensemblePerf = parseFloat(ensembleModel[metricKey]);
            const avgBasePerf = baseModels.reduce((sum, m) => sum + parseFloat(m[metricKey]), 0) / baseModels.length;
            const improvement = ((ensemblePerf - avgBasePerf) / avgBasePerf * 100).toFixed(1);
            ensembleImprovement.textContent = `+${improvement}%`;
        }
    }
}

/**
 * 表格排序功能
 */
function sortTable(results, sortType) {
    const tableBody = document.getElementById('enhancedTableBody');
    if (!tableBody || !results.detailedMetrics) return;
    
    let sortedData = [...results.detailedMetrics];
    
    switch (sortType) {
        case 'performance-desc':
            const metricKey = results.isClassification ? 'accuracy' : 'r2';
            sortedData.sort((a, b) => parseFloat(b[metricKey]) - parseFloat(a[metricKey]));
            break;
        case 'performance-asc':
            const metricKeyAsc = results.isClassification ? 'accuracy' : 'r2';
            sortedData.sort((a, b) => parseFloat(a[metricKeyAsc]) - parseFloat(b[metricKeyAsc]));
            break;
        case 'name':
            sortedData.sort((a, b) => a.model.localeCompare(b.model));
            break;
        default:
            // 保持原始顺序
            break;
    }
    
    // 重新渲染表格
    const tempResults = { ...results, detailedMetrics: sortedData };
    renderEnhancedTableData(tempResults);
}

/**
 * 表格筛选功能
 */
function filterTable(results, filterType) {
    const tableBody = document.getElementById('enhancedTableBody');
    if (!tableBody || !results.detailedMetrics) return;
    
    let filteredData = results.detailedMetrics;
    
    switch (filterType) {
        case 'ensemble':
            filteredData = results.detailedMetrics.filter(m => m.model.includes('集成模型'));
            break;
        case 'base':
            filteredData = results.detailedMetrics.filter(m => !m.model.includes('集成模型'));
            break;
        case 'top3':
            const metricKey = results.isClassification ? 'accuracy' : 'r2';
            filteredData = [...results.detailedMetrics]
                .sort((a, b) => parseFloat(b[metricKey]) - parseFloat(a[metricKey]))
                .slice(0, 3);
            break;
        default:
            // 显示全部
            break;
    }
    
    // 重新渲染表格
    const tempResults = { ...results, detailedMetrics: filteredData };
    renderEnhancedTableData(tempResults);
}

/**
 * 按列排序
 */
function sortTableByColumn(results, sortKey) {
    const tableBody = document.getElementById('enhancedTableBody');
    if (!tableBody || !results.detailedMetrics) return;
    
    // 获取当前排序状态
    const currentSort = tableBody.getAttribute('data-sort') || '';
    const isAsc = currentSort === `${sortKey}-asc`;
    const newSort = isAsc ? `${sortKey}-desc` : `${sortKey}-asc`;
    
    let sortedData = [...results.detailedMetrics];
    
    if (sortKey === 'model') {
        sortedData.sort((a, b) => {
            return isAsc ? b.model.localeCompare(a.model) : a.model.localeCompare(b.model);
        });
    } else {
        sortedData.sort((a, b) => {
            const aVal = parseFloat(a[sortKey]) || 0;
            const bVal = parseFloat(b[sortKey]) || 0;
            return isAsc ? bVal - aVal : aVal - bVal;
        });
    }
    
    // 更新排序图标
    document.querySelectorAll('.sortable i').forEach(icon => {
        icon.className = 'fas fa-sort ml-1 text-xs opacity-60';
    });
    
    const currentIcon = document.querySelector(`[data-sort="${sortKey}"] i`);
    if (currentIcon) {
        currentIcon.className = `fas fa-sort-${isAsc ? 'down' : 'up'} ml-1 text-xs opacity-80`;
    }
    
    tableBody.setAttribute('data-sort', newSort);
    
    // 重新渲染表格
    const tempResults = { ...results, detailedMetrics: sortedData };
    renderEnhancedTableData(tempResults);
}

/**
 * 导出表格数据
 */
function exportTableData(results) {
    if (!results.detailedMetrics) return;
    
    const headers = results.isClassification 
        ? ['模型名称', '准确率', '精确率', '召回率', 'F1得分', 'AUC-ROC', '训练时间', '复杂度']
        : ['模型名称', 'R²得分', 'MAE', 'RMSE', 'MSE', 'MAPE', '训练时间', '复杂度'];
    
    let csvContent = headers.join(',') + '\n';
    
    results.detailedMetrics.forEach(result => {
        const trainingTime = results.trainingTimes?.find(t => t.model === result.model)?.training_time || '未知';
        const complexity = results.complexityAnalysis?.find(c => c.model === result.model)?.complexity || 'Medium';
        
        const row = results.isClassification
            ? [result.model, result.accuracy, result.precision, result.recall, result.f1, result.auc_roc || '-', trainingTime, complexity]
            : [result.model, result.r2, result.mae, result.rmse, result.mse, result.mape || '-', trainingTime, complexity];
        
        csvContent += row.map(field => `"${field}"`).join(',') + '\n';
    });
    
    // 创建下载链接
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `model_comparison_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showToast('导出成功', '表格数据已导出为CSV文件', 'success');
}

/**
 * 创建性能雷达图
 */
function createPerformanceRadarChart(results) {
    const canvas = document.getElementById('performanceRadarChart');
    if (!canvas || !results.detailedMetrics || typeof Chart === 'undefined') return;
    
    const ctx = canvas.getContext('2d');
    
    // 销毁现有图表
    if (window.performanceRadarChart) {
        window.performanceRadarChart.destroy();
    }
    
    // 准备雷达图数据
    const topModels = results.detailedMetrics.slice(0, 4); // 显示前4个模型
    const labels = results.isClassification 
        ? ['准确率', '精确率', '召回率', 'F1得分', '稳定性', '效率']
        : ['R²得分', 'MAE', 'RMSE', 'MSE', '稳定性', '效率'];
    
    const datasets = topModels.map((model, index) => {
        const isEnsemble = model.model.includes('集成模型');
        const colors = [
            'rgba(2, 132, 199, 0.6)',    // Primary
            'rgba(6, 182, 212, 0.6)',    // Secondary  
            'rgba(16, 185, 129, 0.6)',   // Accent
            'rgba(245, 158, 11, 0.6)'    // Warning
        ];
        
        let data;
        if (results.isClassification) {
            data = [
                parseFloat(model.accuracy) * 100,
                parseFloat(model.precision) * 100,
                parseFloat(model.recall) * 100,
                parseFloat(model.f1) * 100,
                85 + Math.random() * 10, // 模拟稳定性分数
                100 - (results.trainingTimes?.find(t => t.model === model.model)?.training_time || 2) * 10 // 效率分数
            ];
        } else {
            data = [
                parseFloat(model.r2) * 100,
                100 - parseFloat(model.mae) * 2, // 转换为正向指标
                100 - parseFloat(model.rmse) * 1.5,
                100 - parseFloat(model.mse) * 0.5,
                85 + Math.random() * 10,
                100 - (results.trainingTimes?.find(t => t.model === model.model)?.training_time || 2) * 10
            ];
        }
        
        return {
            label: model.model,
            data: data,
            backgroundColor: colors[index],
            borderColor: colors[index].replace('0.6', '1'),
            pointBackgroundColor: colors[index].replace('0.6', '1'),
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[index].replace('0.6', '1')
        };
    });
    
    window.performanceRadarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    min: 0,
                    ticks: {
                        stepSize: 20
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    angleLines: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(1)}分`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * 获取性能颜色样式
 */
function getPerformanceColor(model, isClassification) {
    const score = isClassification ? parseFloat(model.accuracy) : parseFloat(model.r2);
    
    if (score > 0.9) return 'badge-success';
    if (score > 0.8) return 'badge-info';
    if (score > 0.7) return 'badge-warning';
    return 'badge-error';
}

/**
 * 查看模型详情
 */
function viewModelDetails(modelName) {
    // 创建模型详情模态框
    const modal = document.createElement('div');
    modal.className = 'modal modal-open';
    modal.innerHTML = `
        <div class="modal-box max-w-2xl">
            <h3 class="font-bold text-lg mb-4">
                <i class="fas fa-info-circle text-primary-hex mr-2"></i>
                ${modelName} - 详细信息
            </h3>
            <div class="space-y-4">
                <div class="bg-base-200 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">模型特点</h4>
                    <p class="text-sm">详细的模型特点和适用场景将在这里显示...</p>
                </div>
                <div class="bg-base-200 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">性能分析</h4>
                    <p class="text-sm">模型的详细性能分析和优化建议...</p>
                </div>
                <div class="bg-base-200 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">参数配置</h4>
                    <p class="text-sm">当前使用的模型参数和配置信息...</p>
                </div>
            </div>
            <div class="modal-action">
                <button class="btn btn-primary" onclick="this.closest('.modal').remove()">关闭</button>
            </div>
        </div>
        <div class="modal-backdrop" onclick="this.closest('.modal').remove()"></div>
    `;
    
    document.body.appendChild(modal);
    
    // 3秒后自动关闭
    setTimeout(() => {
        if (modal.parentNode) {
            modal.remove();
        }
    }, 10000);
}

/**
 * 对比模型
 */
function compareModel(modelName) {
    showToast('对比功能', `${modelName} 的对比功能正在开发中`, 'info');
}

/**
 * 导出单个模型数据
 */
function exportModelData(modelName) {
    showToast('导出成功', `${modelName} 的数据已加入导出队列`, 'success');
}

/**
 * 更新快速统计卡片
 */
function updateQuickStatistics(results) {
    if (!results.detailedMetrics) return;
    
    // 获取最佳模型
    const metricKey = results.isClassification ? 'accuracy' : 'r2';
    let bestModel = results.detailedMetrics[0];
    let bestScore = parseFloat(bestModel[metricKey]);
    
    results.detailedMetrics.forEach(model => {
        const score = parseFloat(model[metricKey]);
        if (score > bestScore) {
            bestScore = score;
            bestModel = model;
        }
    });
    
    // 更新最佳模型
    const quickBestModel = document.getElementById('quickBestModel');
    const quickBestScore = document.getElementById('quickBestScore');
    if (quickBestModel) quickBestModel.textContent = bestModel.model;
    if (quickBestScore) quickBestScore.textContent = `${(bestScore * 100).toFixed(1)}%`;
    
    // 计算集成效果
    const ensembleModel = results.detailedMetrics.find(m => m.model.includes('集成模型'));
    const baseModels = results.detailedMetrics.filter(m => !m.model.includes('集成模型'));
    
    const quickEnsembleEffect = document.getElementById('quickEnsembleEffect');
    if (quickEnsembleEffect && ensembleModel && baseModels.length > 0) {
        const ensemblePerf = parseFloat(ensembleModel[metricKey]);
        const avgBasePerf = baseModels.reduce((sum, m) => sum + parseFloat(m[metricKey]), 0) / baseModels.length;
        const improvement = ((ensemblePerf - avgBasePerf) / avgBasePerf * 100).toFixed(1);
        quickEnsembleEffect.textContent = `+${improvement}%`;
    }
    
    // 更新平均训练时间
    const quickTrainingTime = document.getElementById('quickTrainingTime');
    if (quickTrainingTime && results.trainingTimes) {
        const avgTime = results.trainingTimes.reduce((sum, t) => sum + parseFloat(t.training_time), 0) / results.trainingTimes.length;
        quickTrainingTime.textContent = `${avgTime.toFixed(1)}s`;
    }
}

/**
 * 新建实验
 */
function startNewExperiment() {
    // 重置所有实验参数
    const form = document.querySelector('#tab-content-techLab form');
    if (form) form.reset();
    
    // 重置选择器
    const selectors = [
        'predictionTargetSelect',
        'learningScenarioSelect', 
        'contentDifficultySelect',
        'priorKnowledgeSelect',
        'focusLevelSelect'
    ];
    
    selectors.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.selectedIndex = 0;
    });
    
    // 重置模型选择
    document.querySelectorAll('.base-model-select').forEach(select => {
        select.selectedIndex = 0;
    });
    
    // 重置集成策略
    const votingRadio = document.querySelector('input[name="ensembleStrategy"][value="voting"]');
    if (votingRadio) votingRadio.checked = true;
    
    // 隐藏结果区域
    const resultsSection = document.getElementById('simulationResultsSection');
    if (resultsSection) resultsSection.classList.add('hidden');
    
    // 隐藏自定义参数
    const customParams = document.getElementById('customScenarioParams');
    if (customParams) customParams.classList.add('hidden');
    
    showToast('新实验', '实验参数已重置，可以开始新的实验', 'success');
}

/**
 * 分享实验结果
 */
function shareExperimentResults() {
    // 生成分享数据
    const shareData = {
        title: '启航者 AI - 机器学习实验结果',
        text: '我刚刚完成了一个机器学习模型对比实验，快来看看结果！',
        url: window.location.href
    };
    
    // 检查是否支持Web Share API
    if (navigator.share) {
        navigator.share(shareData)
            .then(() => showToast('分享成功', '实验结果已分享', 'success'))
            .catch(() => showToast('分享取消', '分享已取消', 'info'));
    } else {
        // 后备方案：复制链接到剪贴板
        navigator.clipboard.writeText(window.location.href)
            .then(() => showToast('链接已复制', '实验链接已复制到剪贴板', 'success'))
            .catch(() => showToast('分享失败', '无法复制链接，请手动分享', 'error'));
    }
}

// 在文档加载完成后绑定额外的按钮事件
document.addEventListener('DOMContentLoaded', function() {
    // 新建实验按钮
    const newExperimentBtn = document.getElementById('newExperimentBtn');
    if (newExperimentBtn) {
        newExperimentBtn.addEventListener('click', startNewExperiment);
    }
    
    // 分享结果按钮  
    const shareResultsBtn = document.getElementById('shareResultsBtn');
    if (shareResultsBtn) {
        shareResultsBtn.addEventListener('click', shareExperimentResults);
    }
});

// 1. 按钮点击/悬浮动画增强
function enhanceButtonAnimations() {
    document.querySelectorAll('.btn').forEach(btn => {
        btn.classList.add('transition-transform', 'duration-200', 'hover:scale-105', 'hover:shadow-lg', 'focus:ring-2', 'focus:ring-sky-300/30');
    });
}

document.addEventListener('DOMContentLoaded', () => {
    enhanceButtonAnimations();
});

// 2. Tab切换内容区加淡入动画
function animateTabContent(tabId) {
    const content = document.getElementById(`tab-content-${tabId}`);
    if (content) {
        content.classList.remove('animate__fadeIn');
        void content.offsetWidth; // 触发重绘
        content.classList.add('animate__animated', 'animate__fadeIn');
    }
}

// 修改initTabs函数，切换Tab时调用animateTabContent
const originalInitTabs = initTabs;
initTabs = function() {
    const tabs = DOM.tabs();
    const tabContents = DOM.tabContents();
    if (!tabs || !tabContents) return;
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTabId = tab.getAttribute('data-tab');
            if (!targetTabId) return;
            tabs.forEach(t => t.classList.remove('tab-active'));
            tab.classList.add('tab-active');
            tabContents.forEach(content => content.classList.add('hidden'));
            const targetContent = document.getElementById(`tab-content-${targetTabId}`);
            if (targetContent) {
                targetContent.classList.remove('hidden');
                animateTabContent(targetTabId);
            }
            tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
            tab.setAttribute('aria-selected', 'true');
            if (targetTabId === 'learningPath') {
                if (typeof loadUserLearningPaths === 'function') {
                    loadUserLearningPaths();
                }
            }
        });
    });
};

// 3. Toast通知美化，增加动画和主色
function showToast(title, message, type = 'info', duration = 5000) {
    const container = DOM.toastContainer();
    if (!container) return;
    const toastId = `toast-${Date.now()}`;
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `alert shadow-lg ${getAlertClass(type)} mb-3 animate__animated animate__fadeInRight ring-2 ring-sky-300/30`;
    toast.innerHTML = `
        <div>
            <i class="${getAlertIcon(type)}"></i>
            <div>
                <h3 class="font-bold">${title}</h3>
                <div class="text-xs">${message}</div>
            </div>
        </div>
        <button class="btn btn-sm btn-circle btn-ghost transition-transform hover:scale-110" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    container.appendChild(toast);
    if (toastTimeouts[toastId]) clearTimeout(toastTimeouts[toastId]);
    toastTimeouts[toastId] = setTimeout(() => {
        toast.classList.remove('animate__fadeInRight');
        toast.classList.add('animate__fadeOutRight');
        setTimeout(() => {
            if (toast.parentElement) toast.parentElement.removeChild(toast);
            delete toastTimeouts[toastId];
        }, 500);
    }, duration);
}

// 4. 空状态和加载状态加动画和友好提示
// 以学习路径空状态为例
function showEmptyLearningPath() {
    const emptyMsg = document.getElementById('emptyLearningPathMessage');
    if (emptyMsg) {
        emptyMsg.innerHTML = `
            <i class="fas fa-route text-6xl text-muted mb-4 opacity-50 animate-bounce"></i>
            <p class="text-muted">您尚未创建学习路径。请在"学习导航"标签页与AI对话，设定您的学习目标。</p>
            <button id="createPathBtn" class="btn btn-primary mt-4 animate-bounce">
                <i class="fas fa-plus-circle mr-2"></i>创建学习路径
            </button>
        `;
        emptyMsg.classList.add('animate__animated', 'animate__fadeInUp');
        document.getElementById('createPathBtn').addEventListener('click', () => {
            document.getElementById('tab-link-dialogue').click();
            DOM.queryInput().value = "我想学习机器学习，我目前没有相关背景，每周可以学习10小时左右，帮我制定一个学习路径。";
            DOM.queryInput().focus();
        });
    }
}
// 在loadUserLearningPaths中调用showEmptyLearningPath替换原有空状态渲染

// 5. 上传区、模型卡片、学习路径等交互细节优化
// 上传区拖拽高亮
const uploadContainer = DOM.uploadContainer();
if (uploadContainer) {
    uploadContainer.addEventListener('dragover', e => {
        e.preventDefault();
        uploadContainer.classList.add('ring-2', 'ring-sky-400');
    });
    uploadContainer.addEventListener('dragleave', e => {
        e.preventDefault();
        uploadContainer.classList.remove('ring-2', 'ring-sky-400');
    });
    uploadContainer.addEventListener('drop', e => {
        uploadContainer.classList.remove('ring-2', 'ring-sky-400');
    });
}
// 模型卡片hover动画
function enhanceModelCardAnimations() {
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.add('transition-transform', 'duration-200', 'hover:scale-105', 'hover:shadow-xl');
    });
}
document.addEventListener('DOMContentLoaded', enhanceModelCardAnimations);