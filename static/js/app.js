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
    // Advanced Tools Endpoints
    MODEL_VERSIONS: '/api/ml/model_versions', // POST to create version
    GET_MODEL_VERSIONS: '/api/ml/model_versions/', // GET for list. Append model_name, e.g., /api/ml/model_versions/my_model
    COMPARE_MODELS: '/api/ml/compare_models', // POST
    BUILD_ENSEMBLE: '/api/ml/ensemble', // POST
    DEPLOY_MODEL: '/api/ml/deploy', // POST to deploy
    DEPLOYED_MODELS: '/api/ml/deployments', // GET for list
    UNDEPLOY_MODEL: '/api/ml/undeploy/', // POST. Append deployment_id, e.g., /api/ml/undeploy/deployment_id_123
};

// 模型类别分组，便于前端展示和选择 (与后端 ml_models.py 保持一致)
const MODEL_CATEGORIES = {
    "regression": ["linear_regression", "random_forest_regressor"],
    "classification": ["logistic_regression", "knn_classifier", "decision_tree", "svm_classifier", "naive_bayes", "random_forest_classifier"],
    "clustering": ["kmeans"],
    "ensemble": ["voting_classifier", "voting_regressor", "stacking_classifier", "stacking_regressor", "bagging_classifier", "bagging_regressor"]
};

// 固定模型详细信息
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

// 获取模型图标颜色（可根据需要扩展）
function getModelIconColor(modelInternalName) {
    const colors = {
        "linear_regression": "var(--primary-hex)",       // Sky-600
        "logistic_regression": "var(--secondary-hex)",   // Cyan-500
        "knn_classifier": "#A855F7",                   // Purple-500
        "decision_tree": "var(--accent-hex)",          // Emerald-500
        "svm_classifier": "#EF4444",                   // Red-500
        "naive_bayes": "#6366F1",                   // Indigo-500
        "kmeans": "#EAB308"                        // Yellow-500
    };
    return colors[modelInternalName] || 'var(--neutral-content-hex)'; // Default color
}

// 获取模型类别显示名称
function getCategoryDisplayName(key) {
    if (!key) return '其他';
    const map = {
        'regression': '回归模型',
        'classification': '分类模型',
        'clustering': '聚类模型',
        'ensemble': '集成模型'
    };
    return map[key.toLowerCase()] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
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
    featureImportanceMessage: () => document.getElementById('featureImportanceMessage'),
    confusionMatrixTableContainer: () => document.getElementById('confusionMatrixTableContainer'),
    confusionMatrixMessage: () => document.getElementById('confusionMatrixMessage'),
    modelMetricsContainer: () => document.getElementById('modelMetricsContainer'),
    modelMetricsMessage: () => document.getElementById('modelMetricsMessage'),
    rocCurveChart: () => document.getElementById('rocCurveChart'),
    rocCurveMessage: () => document.getElementById('rocCurveMessage'),
    resultDataTableContainer: () => document.getElementById('resultDataTableContainer'),
    resultDataTableMessage: () => document.getElementById('resultDataTableMessage'),
    particlesBg: () => document.getElementById('particles-js-bg'),
    // Advanced Tools
    versionModelSelector: () => document.getElementById('versionModelSelector'),
    createVersionBtn: () => document.getElementById('createVersionBtn'),
    versionTableBody: () => document.getElementById('versionTableBody'),
    versionMetadataForm: () => document.getElementById('versionMetadataForm'),
    versionFormModelName: () => document.getElementById('versionFormModelName'),
    versionDescription: () => document.getElementById('versionDescription'),
    versionPerformance: () => document.getElementById('versionPerformance'),
    cancelSaveVersionBtn: () => document.getElementById('cancelSaveVersionBtn'),
    saveVersionBtn: () => document.getElementById('saveVersionBtn'),
    compareModelsContainer: () => document.getElementById('compareModelsContainer'),
    addCompareModelBtn: () => document.querySelector('.add-compare-model-btn'),
    compareTestDataSelect: () => document.getElementById('compareTestDataSelect'),
    compareTargetColumnSelect: () => document.getElementById('compareTargetColumnSelect'),
    startCompareBtn: () => document.getElementById('startCompareBtn'),
    compareResultsContainer: () => document.getElementById('compareResultsContainer'),
    compareModelPlaceholder: () => document.querySelector('.compare-model-placeholder'),
    ensembleModelsContainer: () => document.getElementById('ensembleModelsContainer'),
    addEnsembleModelBtn: () => document.querySelector('.add-ensemble-model-btn'),
    ensembleTypeSelect: () => document.getElementById('ensembleTypeSelect'),
    ensembleName: () => document.getElementById('ensembleName'),
    buildEnsembleBtn: () => document.getElementById('buildEnsembleBtn'),
    ensembleResultContainer: () => document.getElementById('ensembleResultContainer'),
    ensembleModelPlaceholder: () => document.querySelector('.ensemble-model-placeholder'),
    deployedModelCount: () => document.getElementById('deployedModelCount'),
    avgResponseTime: () => document.getElementById('avgResponseTime'),
    predictionRequests: () => document.getElementById('predictionRequests'),
    deployModelSelect: () => document.getElementById('deployModelSelect'),
    deployEnvironmentSelect: () => document.getElementById('deployEnvironmentSelect'),
    deployEndpoint: () => document.getElementById('deployEndpoint'),
    deployModelBtn: () => document.getElementById('deployModelBtn'),
    deploymentTableBody: () => document.getElementById('deploymentTableBody'),
    refreshMonitorBtn: () => document.getElementById('refreshMonitorBtn'),
};

/**
 * 主初始化函数
 */
/**
 * HTML特殊字符转义
 * @param {string} unsafe - 需要转义的字符串
 * @returns {string} 转义后的字符串
 */
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

/**
 * 创建并显示模型详情提示框
 * @param {HTMLElement} cardElement - 触发提示框的卡片元素
 * @param {object} modelData - 模型数据，包含 display_name 和 description
 */
function showModelTooltip(cardElement, modelData) {
    if (!modelTooltipElement) {
        modelTooltipElement = document.createElement('div');
        // 样式参考图片中的蓝色提示框
        modelTooltipElement.className = 'fixed z-[100] p-3 bg-blue-600 text-white rounded-md shadow-xl text-xs max-w-xs transition-opacity duration-150 opacity-0 pointer-events-none';
        document.body.appendChild(modelTooltipElement);
    }

    modelTooltipElement.innerHTML = `<p class=\"font-semibold mb-1\">${escapeHtml(modelData.display_name)}</p><p style=\"white-space: pre-wrap;\">${escapeHtml(modelData.description || '暂无详细描述。')}</p>`;

    const cardRect = cardElement.getBoundingClientRect();
    
    // 先让tooltip可见但屏幕外，以便获取其尺寸
    modelTooltipElement.style.visibility = 'hidden';
    modelTooltipElement.style.display = 'block';
    const tooltipRect = modelTooltipElement.getBoundingClientRect();
    modelTooltipElement.style.display = 'none'; // 隐藏以避免闪烁
    modelTooltipElement.style.visibility = 'visible';

    let top = cardRect.top - tooltipRect.height - 8; // 提示框在卡片上方，间隔8px
    let left = cardRect.left + (cardRect.width / 2) - (tooltipRect.width / 2); // 水平居中对齐卡片

    // 边界检查和调整
    if (top < 8) { // 如果提示框顶部超出屏幕
        top = cardRect.bottom + 8; // 移到卡片下方
        // 如果下方空间也不够，尝试放在侧边（这里简化，优先上下）
    }
    if (left < 8) {
        left = 8; // 防止左侧超出屏幕
    }
    if (left + tooltipRect.width > window.innerWidth - 8) {
        left = window.innerWidth - tooltipRect.width - 8; // 防止右侧超出屏幕
    }

    modelTooltipElement.style.top = `${top + window.scrollY}px`; // 加上滚动偏移
    modelTooltipElement.style.left = `${left + window.scrollX}px`; // 加上滚动偏移
    modelTooltipElement.style.opacity = '1';
    modelTooltipElement.classList.remove('pointer-events-none');
}

/**
 * 隐藏模型详情提示框
 */
function hideModelTooltip() {
    if (modelTooltipElement) {
        modelTooltipElement.style.opacity = '0';
        modelTooltipElement.classList.add('pointer-events-none'); 
    }
}

async function main() { // Make main async
    initTabs();
    initUploadToggle();
    initUploadForm();
    initQuerySubmission();
    initExampleQueries();
    initModelSelectionDelegation();
    await loadAvailableModels(); // Wait for models to load before initializing advanced tools
    initDataUploadShortcut();
    initParticlesJS();
    updateQueryInputState();
    initVisualizationTabs();
    initAdvancedTools(); // Initialize advanced tools interactions

    showToast('欢迎使用 AI 机器学习助手 Pro！', 'info', 5000);
}

/**
 * 初始化主导航标签页切换
 */
function initTabs() {
    const tabs = DOM.tabs();
    const tabContents = DOM.tabContents();
    if (!tabs.length || !tabContents.length) return;

    tabs.forEach(tab => {
        tab.addEventListener('click', function(event) {
            event.preventDefault();
            const clickedTab = event.currentTarget;

            tabs.forEach(t => t.classList.remove('tab-active', 'font-semibold'));
            clickedTab.classList.add('tab-active', 'font-semibold');

            tabContents.forEach(content => content.classList.add('hidden'));
            const tabName = clickedTab.getAttribute('data-tab');
            const activeContent = document.getElementById(`tab-content-${tabName}`);
            if (activeContent) {
                activeContent.classList.remove('hidden');
            } else {
                console.error(`Tab content for '${tabName}' not found.`);
            }
        });
    });
}

/**
 * 初始化上传区域展开/折叠
 */
function initUploadToggle() {
    const toggleBtn = DOM.toggleUploadBtn();
    const uploadContainer = DOM.uploadContainer();
    if (!toggleBtn || !uploadContainer) return;

    toggleBtn.addEventListener('click', () => {
        const isHidden = uploadContainer.classList.contains('hidden');
        uploadContainer.classList.toggle('hidden', !isHidden);
        toggleBtn.setAttribute('aria-expanded', isHidden.toString());
        toggleBtn.innerHTML = isHidden ?
            `<i class="fas fa-times mr-2" aria-hidden="true"></i> 关闭上传区域` :
            `<i class="fas fa-file-upload mr-2" aria-hidden="true"></i> 上传新数据`;
    });
}

/**
 * 重置数据分析相关UI和状态
 */
function resetDataAnalysisState() {
    currentData = { path: null, fileName: null, columns: [], columnTypes: {}, rowCount: 0, columnCount: 0, preview: [], analysisCompleted: false };
    selectedTargetColumn = null;
    DOM.dataPreview().innerHTML = '<p class="text-muted p-4 text-center">上传数据文件后将在此显示预览。</p>';
    DOM.dataAnalysisResults().classList.add('hidden');
    DOM.rowCount().textContent = '-';
    DOM.columnCount().textContent = '-';
    DOM.recommendedModels().innerHTML = '-';
    DOM.targetColumnSelector().innerHTML = '<p class="text-muted text-sm p-2 w-full">分析数据后，可选列将在此显示。</p>';
    updateQueryInputState();
}

/**
 * 根据后端推荐的模型列表高亮显示模型卡片
 * @param {string[]} recommendedModels - 后端推荐的模型名称列表
 */
function highlightRecommendedModels(recommendedModels) {
    const grid = DOM.modelGrid();
    if (!grid || !recommendedModels || recommendedModels.length === 0) return;

    grid.querySelectorAll('.model-card').forEach(card => {
        const modelName = card.getAttribute('data-model-name');
        if (recommendedModels.includes(modelName)) {
            card.classList.add('recommended-model-card');
        } else {
            card.classList.remove('recommended-model-card'); // Remove if previously highlighted
        }
    });
}

/**
 * 显示数据预览表格
 */

/**
 * 初始化数据上传表单
 */
function initUploadForm() {
    const form = DOM.uploadForm();
    const fileInput = DOM.dataFile();
    const analyzeBtn = DOM.analyzeDataBtn();
    if (!form || !fileInput || !analyzeBtn) return;

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) {
            showToast('请先选择一个数据文件。', 'error'); return;
        }
        const file = fileInput.files[0];
        // Restrict upload for ML analysis to CSV/Excel as per backend /api/ml/upload
        if (!file.name.toLowerCase().endsWith('.csv') && !file.name.toLowerCase().endsWith('.xlsx') && !file.name.toLowerCase().endsWith('.xls')) {
            showToast('文件格式不受支持。当前仅支持CSV、Excel (.xlsx/.xls) 文件进行数据分析上传。', 'error');
            return;
        }
        const formData = new FormData();
        formData.append('file', file);

        resetDataAnalysisState();
        currentData.fileName = file.name;
        setButtonLoading(analyzeBtn, true, '上传并分析中...');

        try {
            const response = await fetch(API_ENDPOINTS.UPLOAD, { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json().catch(() => ({ error: `HTTP错误: ${response.status}` }));
                throw new Error(err.error);
            }
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            currentData.path = data.file_path;
            currentData.columns = data.columns || [];
            currentData.preview = data.preview || [];
            showDataPreview(currentData.preview, currentData.columns);
            showToast(`文件 "${escapeHtml(currentData.fileName)}" 上传成功! 开始分析...`, 'success');
            await performDataAnalysis(currentData.path);
        } catch (error) {
            console.error('上传或分析错误:', error);
            showToast(`处理失败: ${error.message}`, 'error');
            resetDataAnalysisState(); // Ensure UI reflects failure
        } finally {
            setButtonLoading(analyzeBtn, false, '<i class="fas fa-cogs"></i>分析数据');
            fileInput.value = ''; // Clear file input
        }
    });
}

/**
 * 执行数据分析
 */
async function performDataAnalysis(dataPath) {
    DOM.dataAnalysisResults().classList.remove('hidden');
    ['rowCount', 'columnCount', 'recommendedModels'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<span class="loading loading-dots loading-xs"></span>`;
    });

    try {
        const response = await fetch(API_ENDPOINTS.ANALYZE, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_path: dataPath }),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: `HTTP错误: ${response.status}` }));
            throw new Error(err.error);
        }
        const responseText = await response.text();
        // 处理可能的NaN值
        const cleanedResponse = responseText
            .replace(/:\s*NaN\s*(,|\})/g, ':null$1')
            .replace(/"RK_CITY":\s*NaN/g, '"RK_CITY":null');
        const analysisData = JSON.parse(cleanedResponse, (key, value) => {
            if (typeof value === 'number' && isNaN(value)) return null;
            if (value === 'NaN') return null;
            return value;
        });
        if (analysisData.error) throw new Error(analysisData.error);

        currentData.rowCount = analysisData.row_count || 0;
        currentData.columnCount = analysisData.column_count || 0;
        currentData.columns = analysisData.columns || currentData.columns;
        currentData.columnTypes = analysisData.column_types || {};
        currentData.analysisCompleted = true;

        DOM.rowCount().textContent = currentData.rowCount;
        DOM.columnCount().textContent = currentData.columnCount;
        DOM.recommendedModels().innerHTML = (analysisData.recommended_models && analysisData.recommended_models.length > 0) ?
            analysisData.recommended_models.map(modelKey => `<span class="badge badge-outline badge-primary">${getModelDisplayName(modelKey)}</span>`).join(' ') :
            '无特定推荐';

        createTargetColumnSelector(currentData.columns, currentData.columnTypes);
        highlightRecommendedModels(analysisData.recommended_models);
        showToast('数据分析完成！请选择目标列和模型。', 'success');
    } catch (error) {
        console.error('数据分析错误:', error);
        showToast(`数据分析出错: ${error.message}`, 'error');
        DOM.rowCount().textContent = '错误'; DOM.columnCount().textContent = '错误'; DOM.recommendedModels().textContent = '分析失败';
        currentData.analysisCompleted = false; // Mark as not completed on error
    } finally {
        updateQueryInputState();
    }
}

/**
 * 显示数据预览表格
 * 优化表格展示，当列数过多时分行展示
 */
function showDataPreview(previewData, columns) {
    const container = DOM.dataPreview();
    if (!container) return;
    if (previewData && previewData.length > 0 && columns && columns.length > 0) {
        // 判断列数是否过多，需要分行展示
        const MAX_COLUMNS_PER_ROW = 6; // 每行最多显示的列数
        const needMultipleRows = columns.length > MAX_COLUMNS_PER_ROW;
        
        if (needMultipleRows) {
            // 分行展示的表格
            let tableHTML = '';
            // 计算需要多少行来展示所有列
            const rowCount = Math.ceil(columns.length / MAX_COLUMNS_PER_ROW);
            
            for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
                // 获取当前行要显示的列
                const startColIndex = rowIndex * MAX_COLUMNS_PER_ROW;
                const endColIndex = Math.min(startColIndex + MAX_COLUMNS_PER_ROW, columns.length);
                const currentRowColumns = columns.slice(startColIndex, endColIndex);
                
                tableHTML += `<div class="mb-4"><div class="overflow-x-auto custom-scrollbar"><table class="table table-xs sm:table-sm w-full"><thead><tr>`;
                currentRowColumns.forEach(col => tableHTML += `<th>${escapeHtml(col)}</th>`);
                tableHTML += `</tr></thead><tbody>`;
                
                previewData.forEach(row => {
                    tableHTML += `<tr>`;
                    currentRowColumns.forEach(col => tableHTML += `<td>${escapeHtml(row[col] ?? '')}</td>`);
                    tableHTML += `</tr>`;
                });
                
                tableHTML += `</tbody></table></div></div>`;
            }
            
            container.innerHTML = tableHTML;
        } else {
            // 原始单行表格展示
            let tableHTML = `<div class="overflow-x-auto custom-scrollbar"><table class="table table-xs sm:table-sm w-full"><thead><tr>`;
            columns.forEach(col => tableHTML += `<th>${escapeHtml(col)}</th>`);
            tableHTML += `</tr></thead><tbody>`;
            previewData.forEach(row => {
                tableHTML += `<tr>`;
                columns.forEach(col => tableHTML += `<td>${escapeHtml(row[col] ?? '')}</td>`);
                tableHTML += `</tr>`;
            });
            tableHTML += `</tbody></table></div>`;
            container.innerHTML = tableHTML;
        }
    } else {
        container.innerHTML = '<p class="text-muted p-4 text-center">无有效数据可供预览。</p>';
    }
}

/**
 * 创建目标列选择按钮
 */
function createTargetColumnSelector(columns, columnTypes) {
    const container = DOM.targetColumnSelector();
    if (!container) return;
    container.innerHTML = '';
    if (!columns || columns.length === 0) {
        container.innerHTML = '<p class="text-muted text-sm p-2 w-full">无可用列。</p>'; return;
    }
    
    // 创建选择器容器
    const selectWrapper = document.createElement('div');
    selectWrapper.className = 'w-full';
    
    // 创建下拉选择器
    const select = document.createElement('select');
    select.className = 'select select-bordered select-sm w-full';
    select.id = 'targetColumnSelect';
    
    // 添加默认选项
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '请选择目标列';
    defaultOption.selected = true;
    defaultOption.disabled = true;
    select.appendChild(defaultOption);
    
    // 添加列选项
    columns.forEach(column => {
        const type = columnTypes[column] || '未知';
        const option = document.createElement('option');
        option.value = column;
        option.textContent = `${escapeHtml(column)} (${type === 'numerical' ? '数值' : '分类'})`;
        select.appendChild(option);
    });
    
    // 添加事件监听
    select.addEventListener('change', function() {
        if (this.value) {
            selectedTargetColumn = this.value;
            showToast(`已选目标列: "${escapeHtml(this.value)}"`, 'info');
            updateQueryInputState();
        }
    });
    
    selectWrapper.appendChild(select);
    container.appendChild(selectWrapper);
}

/**
 * 加载并显示可用模型
 */
/**
 * 加载并显示可用模型，实现类似图片效果的布局、样式和悬停提示。
 */
/**
 * 异步加载并显示所有可用的机器学习模型。
 * 此函数会从 FIXED_MODEL_DETAILS 获取模型数据，按类别组织它们，
 * 并为每个模型动态创建卡片元素，然后将它们添加到DOM中。
 * 卡片支持点击翻转以显示更多信息，并有悬停提示。
 * 同时会更新模型总数徽章。
 */
async function loadAvailableModels() {
    const gridContainer = DOM.modelGrid();
    const modelCountBadge = DOM.modelCountBadge();

    if (!gridContainer) {
        console.error("Model grid container not found.");
        return;
    }

    gridContainer.innerHTML = ''; // 清空现有内容

    // 使用 FIXED_MODEL_DETAILS 作为数据源，因为它包含了所有必要信息
    const modelsToDisplay = Object.values(FIXED_MODEL_DETAILS);

    if (modelsToDisplay.length === 0) {
        gridContainer.innerHTML = '<p class="text-center text-muted col-span-full p-4">暂无可用模型。</p>';
        if (modelCountBadge) {
            modelCountBadge.textContent = '共 0 个模型';
            modelCountBadge.className = 'badge border-transparent bg-gray-400 text-white py-2 px-3 text-xs sm:text-sm'; // 更新为灰色背景
        }
        return;
    }

    // 更新模型总数徽章样式和内容
    if (modelCountBadge) {
        modelCountBadge.textContent = `共 ${modelsToDisplay.length} 个模型`;
        modelCountBadge.className = 'badge border-transparent bg-purple-500 text-white py-2 px-3 text-xs sm:text-sm'; // 图片中的紫色背景
    }

    // 按类别分组模型
    const categorizedModels = {};
    modelsToDisplay.forEach(model => {
        const categoryKey = getCategoryForModel(model.internal_name) || 'other';
        if (!categorizedModels[categoryKey]) {
            categorizedModels[categoryKey] = [];
        }
        categorizedModels[categoryKey].push(model);
    });

    const fragment = document.createDocumentFragment();

    // 定义类别显示顺序和标题 (与图片一致)
    const categoryDisplayOrder = [
        { key: 'classification', title: '分类模型' },
        { key: 'regression', title: '回归模型' },
        { key: 'clustering', title: '聚类模型' },
        // { key: 'ensemble', title: '集成模型' }, // 图片中未显示集成模型，暂时注释
        // { key: 'other', title: '其他模型' } // 图片中未显示其他模型，暂时注释
    ];

    categoryDisplayOrder.forEach(categoryInfo => {
        const categoryKey = categoryInfo.key;
        const categoryTitleText = categoryInfo.title;

        if (categorizedModels[categoryKey] && categorizedModels[categoryKey].length > 0) {
            const categorySection = document.createElement('div');
            categorySection.className = 'mb-6 model-category-section'; // 类别之间的间距

            const categoryTitleElement = document.createElement('h3');
            // 类别标题样式 (参考图片，可进一步调整)
            categoryTitleElement.className = 'text-xl font-semibold mb-3 text-neutral-content-hex'; 
            categoryTitleElement.textContent = categoryTitleText;
            categorySection.appendChild(categoryTitleElement);

            const categoryGrid = document.createElement('div');
            // 模型卡片网格布局 (参考图片，5列)
            categoryGrid.className = 'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4'; 
            categorySection.appendChild(categoryGrid);

            categorizedModels[categoryKey].forEach(model => {
                /**
                 * @type {HTMLDivElement} card - 模型卡片元素
                 */
                const card = document.createElement('div');
                // 简化类名，主要依赖 index.html 中的 .model-card CSS 定义
                card.className = 'model-card'; 
                card.setAttribute('data-model-name', model.internal_name);
                card.setAttribute('tabindex', '0'); // For accessibility
                card.setAttribute('role', 'button');
                card.setAttribute('aria-label', `选择模型 ${escapeHtml(model.display_name)}`);

                const iconColor = getModelIconColor(model.internal_name);
                const modelCategoryName = getCategoryDisplayName(getCategoryForModel(model.internal_name));

                // 卡片内部结构，包含正面和背面，用于翻转效果
                // Tailwind CSS 类用于布局和基本样式，具体翻转和精细样式在 index.html 中定义
                card.innerHTML = `
                    <div class="model-card-inner">
                        <div class="model-card-front">
                            <i class="fas ${escapeHtml(model.icon_class)}" style="color: ${iconColor};"></i>
                            <h4>${escapeHtml(model.display_name)}</h4>
                            <p>${escapeHtml(modelCategoryName)}</p>
                        </div>
                        <div class="model-card-back">
                            <h4>${escapeHtml(model.display_name)}</h4>
                            <p>${escapeHtml(model.description || '暂无详细描述。')}</p>
                        </div>
                    </div>
                `;
                
                /**
                 * 处理模型卡片点击事件，用于翻转卡片。
                 * 注意：此处的翻转与模型选择是分离的。模型选择在 initModelSelectionDelegation 中处理。
                 * @param {MouseEvent} event - 点击事件对象
                 */
                card.addEventListener('click', (event) => {
                    // 确保点击的是卡片本身或其子元素，而不是其他交互元素（如果未来添加的话）
                    if (event.target.closest('.model-card')) {
                        const currentCard = event.target.closest('.model-card');
                        // 如果卡片已被选中，则不执行翻转，保持正面显示
                        // if (!currentCard.classList.contains('selected-model-card')) {
                        //     currentCard.classList.toggle('is-flipped');
                        // }
                        // 上述翻转逻辑已移至 initModelSelectionDelegation 统一处理，此处不再单独切换翻转状态
                        // 仅当卡片未被选中时，允许通过点击（非选择操作）来翻转查看背面信息
                        // 真正的选择逻辑会确保选中的卡片是正面朝上的
                        if (!currentCard.classList.contains('selected-model-card')) {
                             // 允许用户点击非选中的卡片来查看背面，但这个翻转是临时的
                             // is-flipped 状态会在 initModelSelectionDelegation 中被正确管理
                        }

                    }
                });
                
                // 添加悬停和焦点事件监听器以显示/隐藏模型详情提示框
                card.addEventListener('mouseenter', (event) => showModelTooltip(event.currentTarget, model));
                card.addEventListener('mouseleave', hideModelTooltip);
                card.addEventListener('focus', (event) => showModelTooltip(event.currentTarget, model));
                card.addEventListener('blur', hideModelTooltip);

                categoryGrid.appendChild(card);
            });
            fragment.appendChild(categorySection);
        }
    });

    gridContainer.appendChild(fragment);

    // 初始化高级工具选择器的数据 (如果需要)
    const modelsForAdvancedTools = modelsToDisplay.map(m => ({ 
        name: m.internal_name, 
        type: getCategoryForModel(m.internal_name),
    }));
    // 如果 populateAdvancedToolSelectors 函数存在且需要，则调用
    if (typeof populateAdvancedToolSelectors === 'function') {
        await populateAdvancedToolSelectors(modelsForAdvancedTools);
    }
}

/**
 * 获取模型默认图标
 */
function getDefaultModelIcon(internalName) {
    const name = (internalName || '').toLowerCase();
    if (name.includes('linear')) return 'fa-chart-line';
    if (name.includes('logistic')) return 'fa-code-branch';
    if (name.includes('tree')) return 'fa-sitemap';
    if (name.includes('forest')) return 'fa-tree';
    if (name.includes('knn')) return 'fa-project-diagram';
    if (name.includes('svm')) return 'fa-network-wired';
    if (name.includes('bayes')) return 'fa-percentage';
    if (name.includes('means')) return 'fa-object-group';
    return 'fa-brain';
}

/**
 * 创建模型卡片元素
 */
function createModelCardElement(internalName, displayName, type, description, icon) {
    const card = document.createElement('div');
    card.className = 'model-card h-48 content-card';
    card.setAttribute('data-model-name', internalName);
    card.tabIndex = 0; card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `选择 ${displayName} 模型`);
    card.innerHTML = `
        <div class="model-card-inner h-full">
            <div class="model-card-front">
                <i class="fas ${icon} text-4xl mb-2.5" aria-hidden="true"></i>
                <div class="font-semibold text-base">${escapeHtml(displayName)}</div>
                <div class="text-xs text-muted mt-1.5">${escapeHtml(type)}</div>
            </div>
            <div class="model-card-back">
                <h4 class="font-semibold text-base mb-1.5">${escapeHtml(displayName)}</h4>
                <p class="text-xs opacity-80 mb-2.5 px-1">${escapeHtml(description || '暂无详细描述。')}</p>
                <button type="button" class="btn btn-xs btn-outline select-model-btn">选择此模型</button>
            </div>
        </div>`;
    return card;
}

/**
 * 初始化模型选择（事件委托）
 */
/**
 * 初始化模型选择（事件委托），并更新选中样式以匹配图片。
 */
function initModelSelectionDelegation() {
    const grid = DOM.modelGrid();
    if (!grid) return;

    const handleSelection = (targetCard) => {
        const modelName = targetCard.getAttribute('data-model-name');
        if (!modelName) return;

        // 如果点击的是已选中的卡片，则取消选择 (可选行为)
        // if (selectedModelName === modelName) {
        //     selectedModelName = null;
        //     targetCard.classList.remove('selected-model-card', 'bg-blue-600', 'text-white', 'border-blue-700', 'shadow-xl');
        //     targetCard.classList.add('bg-base-100', 'border-base-300', 'shadow-md');
        //     const icon = targetCard.querySelector('i.fas');
        //     if (icon) icon.style.color = getModelIconColor(modelName); // 恢复原始图标颜色
        //     DOM.selectedModelInfo().textContent = '当前未选择模型。请点击模型卡片进行选择。';
        //     updateQueryInputState();
        //     return;
        // }

        selectedModelName = modelName;

        // 移除所有卡片的选中样式
        grid.querySelectorAll('.model-card').forEach(c => {
            c.classList.remove('selected-model-card', 'bg-blue-600', 'text-white', 'border-blue-700', 'shadow-xl');
            c.classList.remove('is-flipped'); // 确保所有卡片都恢复到正面
            c.classList.add('bg-base-100', 'border-base-300', 'shadow-md'); // 恢复默认样式
            const icon = c.querySelector('i.fas');
            if (icon) {
                const originalColor = c.getAttribute('data-original-icon-color') || getModelIconColor(c.getAttribute('data-model-name'));
                icon.style.color = originalColor;
            }
        });

        // 为选中的卡片添加新样式 (参考图片中的蓝色选中效果)
        targetCard.classList.add('selected-model-card', 'bg-blue-600', 'text-white', 'border-blue-700', 'shadow-xl');
        targetCard.classList.remove('bg-base-100', 'border-base-300', 'shadow-md');
        targetCard.classList.remove('is-flipped'); // 确保选中的卡片正面朝上
        
        const icon = targetCard.querySelector('i.fas');
        if (icon) {
            if (!targetCard.hasAttribute('data-original-icon-color')) {
                 targetCard.setAttribute('data-original-icon-color', icon.style.color);
            }
            icon.style.color = 'white'; // 选中时图标变为白色
        }
        
        const modelDetails = FIXED_MODEL_DETAILS[modelName] || { display_name: modelName };
        DOM.selectedModelInfo().innerHTML = `已选择模型: <strong class="text-blue-600">${escapeHtml(modelDetails.display_name)}</strong>.`;
        showToast(`已选模型: "${escapeHtml(modelDetails.display_name)}"`, 'info');
        updateQueryInputState();

        // 滚动到查询区域（如果模型选择后需要用户立即操作）
        const querySection = document.getElementById('querySection');
        if (querySection) {
            querySection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    };

    grid.addEventListener('click', (e) => {
        const card = e.target.closest('.model-card');
        if (card) {
            handleSelection(card);
        }
    });

    grid.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            const card = document.activeElement.closest('.model-card');
            if (card && grid.contains(card)) {
                e.preventDefault();
                handleSelection(card);
            }
        }
    });
}

/**
 * 更新查询输入框状态
 */
function updateQueryInputState() {
    const input = DOM.queryInput();
    const label = DOM.queryInputLabel();
    const btn = DOM.submitQueryButton();
    const info = DOM.modeSpecificInfo();
    const uploadBtn = DOM.uploadDataShortcutBtn();
    const mode = document.querySelector('input[name="queryMode"]:checked')?.value;
    if (!input || !label || !btn || !info) return;

    let placeholder = '请输入您的问题...', labelText = '您想解决什么问题？', disabled = false, infoText = '';
    
    // 通用大模型模式
    if (mode === 'general_llm') {
        infoText = '通用模式：可直接提问，无需上传数据或选择模型。';
        // 禁用数据上传按钮
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.classList.add('opacity-50', 'cursor-not-allowed');
            uploadBtn.setAttribute('data-tip', '通用大模型模式下不可上传数据');
        }
        // 清除数据和模型选择的提示
        labelText = '您想解决什么问题？';
    } 
    // 数据分析模式
    else {
        infoText = '数据分析模式：将基于您上传的数据和选择的模型进行分析。';
        // 启用数据上传按钮
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            uploadBtn.setAttribute('data-tip', '快速跳转到数据上传');
        }
        
        if (!currentData.path) {
            disabled = true; placeholder = '请先上传数据。'; labelText += ' (需上传数据)'; infoText = '请先上传数据。';
        } else if (!currentData.analysisCompleted) {
            disabled = true; placeholder = '数据分析中，请稍候...'; labelText += ' (分析进行中)'; infoText = '数据正在分析中，完成后即可提问。';
        } else { // Data uploaded and analyzed, allow queries
            placeholder = `基于 ${escapeHtml(currentData.fileName || '已上传数据')} 进行分析或提问...`;
            labelText = `<i class='fas fa-brain mr-2'></i> 基于 ${escapeHtml(currentData.fileName || '已上传数据')} 分析:`;
            infoText = `数据已就绪: ${escapeHtml(currentData.fileName || '已上传数据')}。您现在可以提问了。`;
            // Model and target column are now optional for enabling the query
        }
    }
    input.disabled = disabled; btn.disabled = disabled;
    input.placeholder = placeholder; label.innerHTML = labelText; info.textContent = infoText;
}

/**
 * 初始化查询提交
 */
function initQuerySubmission() {
    const btn = DOM.submitQueryButton();
    const input = DOM.queryInput();
    if (!btn || !input) return;

    DOM.queryModeSelector().forEach(radio => radio.addEventListener('change', updateQueryInputState));
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !input.disabled && !btn.disabled) {
            e.preventDefault(); btn.click();
        }
    });
    btn.addEventListener('click', async () => {
        const query = input.value.trim();
        if (!query) { showToast('请输入查询内容。', 'warning'); return; }
        const mode = document.querySelector('input[name="queryMode"]:checked').value;
        const body = { 
            query,
            mode,
            use_existing_model: true // 默认使用现有模型，除非特定操作需要训练
        };
        
        if (mode === 'data_analysis') {
            if (!currentData.path || !currentData.analysisCompleted) { 
                showToast('请先上传并成功分析数据。', 'error'); 
                setButtonLoading(btn, false, '提交查询', DOM.submitQueryIcon());
                showLoadingSpinner(false);
                return; 
            }
            body.data_path = currentData.path;
            // Model and target column are now optional for the payload
            // They will be included if selected, but not required
            if (selectedModelName) {
                body.model_name = selectedModelName;
            }
            if (selectedTargetColumn) {
                body.target_column = selectedTargetColumn;
            }

            // 添加数据预览（前5行）
            if (currentData.preview && currentData.preview.length > 0) {
                body.data_preview = currentData.preview.slice(0, 5);
            } else {
                body.data_preview = []; // 如果没有预览数据，发送空数组
            }
        }
        setButtonLoading(btn, true, '处理中...', DOM.submitQueryIcon());
        showLoadingSpinner(true, 'AI思考中，请稍候...');
        clearPreviousResults();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 60s timeout
        try {
            const response = await fetch(API_ENDPOINTS.CHAT, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body), signal: controller.signal,
            });
            clearTimeout(timeoutId);
            if (!response.ok) {
                const err = await response.json().catch(() => ({ error: `服务器错误 (${response.status})` }));
                throw new Error(err.error || (response.status === 504 ? '服务器处理超时' : `请求失败 ${response.status}`));
            }
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            displayChatResponse(data, query);
            saveToHistory(query, data, mode);
        } catch (error) {
            console.error('查询错误:', error); clearTimeout(timeoutId);
            let msg = error.name === 'AbortError' ? '请求超时' : error.message;
            let errorHTML = '';
            
            // 根据错误类型提供更友好的错误信息
            if (error.name === 'AbortError') {
                errorHTML = `
                    <div class="alert alert-error shadow-lg mb-4">
                        <div>
                            <i class="fas fa-exclamation-circle"></i>
                            <span>请求超时</span>
                        </div>
                    </div>
                    <p>服务器处理您的请求时间过长。这可能是因为：</p>
                    <ul class="list-disc pl-5 my-3">
                        <li>您的查询过于复杂</li>
                        <li>服务器当前负载较高</li>
                        <li>网络连接问题</li>
                    </ul>
                    <p>建议：尝试简化您的问题，或稍后再试。</p>
                `;
            } else if (msg.startsWith('Could not parse LLM output: ')) {
                 // 提取并显示部分AI输出
                 const partialOutput = msg.substring('Could not parse LLM output: '.length);
                 errorHTML = `
                     <div class="alert alert-warning shadow-lg mb-4">
                         <div>
                             <i class="fas fa-exclamation-triangle"></i>
                             <span>AI处理复杂任务时遇到了限制</span>
                         </div>
                     </div>
                     <p>您的查询过于复杂，AI无法在允许的时间或步骤内完成处理。以下是AI在处理过程中的部分结果：</p>
                     <div class="mt-4 p-4 bg-base-200 rounded-md overflow-auto max-h-60">
                         <pre class="whitespace-pre-wrap">${escapeHtml(partialOutput)}</pre>
                     </div>
                     <p class="mt-4">建议：尝试将您的问题拆分为更小的部分，或者提供更具体的指令。</p>
                 `;
            } else {
                errorHTML = `<p class="text-error">查询失败: ${escapeHtml(msg)}</p>`;
            }
            
            showToast(`查询失败: ${msg}`, 'error');
            DOM.responseText().innerHTML = errorHTML;
        } finally {
            setButtonLoading(btn, false, '提交问题', DOM.submitQueryIcon(), 'fa-paper-plane');
            showLoadingSpinner(false);
            DOM.responseSection().classList.remove('hidden');
        }
    });
}

/**
 * 清除上次结果
 */
function clearPreviousResults() {
    if (!DOM.responseText()) return;
    if (DOM.responseText()) {
    DOM.responseText().innerHTML = '<p class="text-muted">等待AI响应...</p>';
}
    
    // 安全地操作DOM元素
    const visualizationArea = DOM.visualizationDisplayArea();
    const sourceArea = DOM.sourceDocumentsArea();
    const sourceList = DOM.sourceDocumentsList();
    const sourceMsg = DOM.sourceDocumentsMessage();
    const confusionMatrix = DOM.confusionMatrixTableContainer();
    const metricsContainer = DOM.modelMetricsContainer();
    const dataTableContainer = DOM.resultDataTableContainer();
    
    if (visualizationArea) visualizationArea.classList.add('hidden');
    if (sourceArea) sourceArea.classList.add('hidden');
    if (sourceList) sourceList.innerHTML = '';
    if (sourceMsg) sourceMsg.classList.remove('hidden');
    
    ['mainChart', 'featureImportanceChart', 'rocCurveChart'].forEach(id => {
        const msgEl = document.getElementById(`${id}Message`);
        if (msgEl) msgEl.classList.add('hidden');
        if (activeCharts[id]) { 
            activeCharts[id].destroy(); 
            delete activeCharts[id]; 
        }
    });
    
    if (confusionMatrix) confusionMatrix.innerHTML = `<p id="confusionMatrixMessage" class="text-muted p-2 text-sm text-center">无混淆矩阵</p>`;
    if (metricsContainer) metricsContainer.innerHTML = `<p id="modelMetricsMessage" class="text-muted p-2 text-sm text-center">无评估指标</p>`;
    if (dataTableContainer) dataTableContainer.innerHTML = `<p id="resultDataTableMessage" class="text-muted p-5 text-center">无详细数据表</p>`;
}

/**
 * 显示聊天响应
 */
function displayChatResponse(data, userQuery) {
    const resultsTab = document.getElementById('tab-link-results');
    if (resultsTab && !resultsTab.classList.contains('tab-active')) resultsTab.click();
    
    // 检查是否包含代理超时或迭代限制错误
    if (data.answer && (data.answer.includes('Agent stopped due to iteration limit') || 
                        data.answer.includes('time limit'))) {
        // 处理代理超时或迭代限制错误
        if (DOM.responseText()) {
            DOM.responseText().innerHTML = `
                <div class="alert alert-warning shadow-lg mb-4">
                    <div>
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>AI处理复杂任务时遇到了限制</span>
                    </div>
                </div>
                <p class="mb-4">您的查询过于复杂，AI无法在允许的时间或步骤内完成处理。以下是AI在处理过程中的部分结果：</p>
                <div class="p-4 bg-base-200 rounded-lg">${formatAnswer(data.answer.replace('Agent stopped due to iteration limit or time limit.', ''))}</div>
                <p class="mt-4 text-sm text-muted">建议：尝试将您的问题拆分为更小的部分，或者提供更具体的指令。</p>
            `;
            return;
        }
    }
    
    const responseSection = DOM.responseSection();
    const responseText = DOM.responseText();
    const visualizationArea = DOM.visualizationDisplayArea();
    const sourceArea = DOM.sourceDocumentsArea();
    const sourceList = DOM.sourceDocumentsList();
    const sourceMsg = DOM.sourceDocumentsMessage();
    
    if (responseSection) responseSection.classList.remove('hidden');
    if (responseText && data.answer) {
        // 检查是否有其他错误信息需要特殊处理
        if (data.answer.includes('Invalid or incomplete response')) {
            responseText.innerHTML = `
                <div class="alert alert-warning shadow-lg mb-4">
                    <div>
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>AI响应格式不完整</span>
                    </div>
                </div>
                <p class="mb-4">AI生成的响应格式不完整或无效。以下是AI在处理过程中的部分结果：</p>
                <div class="p-4 bg-base-200 rounded-lg">${formatAnswer(data.answer.replace('Invalid or incomplete response', ''))}</div>
            `;
        } else {
            responseText.innerHTML = formatAnswer(data.answer);
        }
    }
    
    if (visualizationArea) {
        if (data.visualization_data && Object.keys(data.visualization_data).length > 0) {
            visualizationArea.classList.remove('hidden');
            renderVisualizations(data.visualization_data);
        } else {
            visualizationArea.classList.add('hidden');
        }
    }
    
    if (sourceArea && sourceList && sourceMsg) {
        if (data.source_documents && data.source_documents.length > 0) {
            sourceArea.classList.remove('hidden');
            sourceMsg.classList.add('hidden');
            sourceList.innerHTML = data.source_documents.map(doc =>
                `<li class="p-2 border border-base-300 rounded-md bg-base-200/50 hover:bg-base-300/50 transition-colors">
                 <strong class="text-primary-focus">${escapeHtml(doc.metadata?.source || '未知来源')}</strong>: 
                 <small class="text-muted">${escapeHtml(doc.page_content?.substring(0, 150) || '')}...</small></li>`
            ).join('');
        } else {
            sourceArea.classList.add('hidden');
            sourceMsg.classList.remove('hidden');
        }
    }
}

/**
 * 渲染可视化数据
 */
function renderVisualizations(vizData) {
    const chartOrDefault = (id, dataObj, defaultMsg) => {
        const msgEl = document.getElementById(`${id}Message`);
        if (dataObj && DOM[id]()) {
            renderChart(id, dataObj.type, dataObj.data, dataObj.options, msgEl);
        } else if (msgEl) {
            msgEl.textContent = defaultMsg || `无${id.replace(/([A-Z])/g, ' $1').toLowerCase()}数据。`;
            msgEl.classList.remove('hidden');
            if (activeCharts[id]) { activeCharts[id].destroy(); delete activeCharts[id]; }
        }
    };
    chartOrDefault('mainResultChart', vizData.mainChart, '无主要图表数据。');
    chartOrDefault('featureImportanceChart', vizData.featureImportance, '无特征重要性数据。');
    chartOrDefault('rocCurveChart', vizData.rocCurve, '无ROC曲线数据 (可能不适用)。');

    if (vizData.confusionMatrix && DOM.confusionMatrixTableContainer()) {
        DOM.confusionMatrixTableContainer().innerHTML = createHtmlTable(vizData.confusionMatrix.data, vizData.confusionMatrix.headers, "混淆矩阵");
    } else if (DOM.confusionMatrixMessage()) {
        DOM.confusionMatrixMessage().textContent = '无混淆矩阵。'; DOM.confusionMatrixMessage().classList.remove('hidden');
    }
    if (vizData.modelMetrics && DOM.modelMetricsContainer()) {
        DOM.modelMetricsContainer().innerHTML = Object.entries(vizData.modelMetrics).map(([key, value]) => `
            <div class="stat p-2 border border-base-200 rounded">
                <div class="stat-title text-xs">${escapeHtml(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))}</div>
                <div class="stat-value text-lg">${typeof value === 'number' ? value.toFixed(4) : escapeHtml(value)}</div>
            </div>`).join('') || `<p id="modelMetricsMessage" class="text-muted p-2 text-sm text-center">无评估指标</p>`;
    } else if (DOM.modelMetricsMessage()) {
        DOM.modelMetricsMessage().textContent = '无评估指标。'; DOM.modelMetricsMessage().classList.remove('hidden');
    }
    if (vizData.dataTable && DOM.resultDataTableContainer()) {
        DOM.resultDataTableContainer().innerHTML = createHtmlTable(vizData.dataTable.data, vizData.dataTable.headers, "详细数据");
    } else if (DOM.resultDataTableMessage()) {
        DOM.resultDataTableMessage().textContent = '无详细数据表。'; DOM.resultDataTableMessage().classList.remove('hidden');
    }
}

/**
 * 创建HTML表格
 */
function createHtmlTable(rows, headers, caption = "数据表") {
    if (!rows || !rows.length || !headers || !headers.length) return `<p class="text-muted p-4 text-center">无数据显示。</p>`;
    let table = `<div class="overflow-x-auto custom-scrollbar"><table class="table table-xs sm:table-sm w-full">`;
    if (caption) table += `<caption>${escapeHtml(caption)}</caption>`;
    table += `<thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead><tbody>`;
    rows.forEach(row => {
        table += `<tr>${headers.map(h => `<td>${escapeHtml(row[h] ?? '')}</td>`).join('')}</tr>`;
    });
    table += `</tbody></table></div>`;
    return table;
}

/**
 * 渲染图表
 */
function renderChart(canvasId, type, data, backendOptions, messageEl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas ${canvasId} not found.`);
        if (messageEl) { messageEl.textContent = `图表渲染错误: 画布未找到。`; messageEl.classList.remove('hidden'); }
        return;
    }
    if (activeCharts[canvasId]) activeCharts[canvasId].destroy();

    const baseFont = "'Noto Sans SC', sans-serif";
    const baseColor = getComputedStyle(document.documentElement).getPropertyValue('--base-content-hex').trim();
    const mutedColor = getComputedStyle(document.documentElement).getPropertyValue('--neutral-content-hex').trim();
    const gridColor = getComputedStyle(document.documentElement).getPropertyValue('--base-300-hex').trim();

    let defaultOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top', labels: { color: baseColor, font: { family: baseFont } } },
            tooltip: { titleFont: { family: baseFont }, bodyFont: { family: baseFont } }
        },
        scales: {
            x: { ticks: { color: mutedColor, font: { family: baseFont } }, grid: { color: gridColor, drawOnChartArea: false } },
            y: { ticks: { color: mutedColor, font: { family: baseFont } }, grid: { color: gridColor, borderDash: [2, 3] }, beginAtZero: true }
        }
    };
    if (type === 'pie' || type === 'doughnut') { delete defaultOpts.scales; defaultOpts.plugins.legend.position = 'right'; }
    if (type === 'radar') defaultOpts.scales = { r: { angleLines: { color: gridColor }, grid: { color: gridColor }, pointLabels: { font: { family: baseFont, size: 10 }, color: mutedColor }, ticks: { backdropColor: 'transparent', color: mutedColor }}};

    // Basic deep merge for options (can be improved with a proper deep merge utility)
    const finalOptions = {
        ...defaultOpts,
        ...(backendOptions || {}),
        plugins: { ...defaultOpts.plugins, ...backendOptions?.plugins,
            legend: {...defaultOpts.plugins?.legend, ...backendOptions?.plugins?.legend},
            tooltip: {...defaultOpts.plugins?.tooltip, ...backendOptions?.plugins?.tooltip}
        },
        scales: { ...defaultOpts.scales, ...backendOptions?.scales,
            x: {...defaultOpts.scales?.x, ...backendOptions?.scales?.x},
            y: {...defaultOpts.scales?.y, ...backendOptions?.scales?.y},
            r: {...defaultOpts.scales?.r, ...backendOptions?.scales?.r} // For radar
        }
    };

    activeCharts[canvasId] = new Chart(canvas.getContext('2d'), { type, data, options: finalOptions });
    if (messageEl) messageEl.classList.add('hidden');
}

/**
 * 初始化可视化结果标签页
 */
function initVisualizationTabs() {
    const tabs = DOM.vizTabs();
    const contents = DOM.vizContents();
    if (!tabs.length || !contents.length) return;
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            tabs.forEach(t => t.classList.remove('tab-active'));
            e.currentTarget.classList.add('tab-active');
            contents.forEach(c => c.classList.add('hidden'));
            document.getElementById(`viz-content-${e.currentTarget.dataset.viz}`)?.classList.remove('hidden');
        });
    });
}

/**
 * 设置按钮加载状态
 */
function setButtonLoading(btn, isLoading, loadingText = '处理中...', iconEl = null, originalIcon = null) {
    if (!btn) return;
    btn.disabled = isLoading;
    const textSpan = btn.querySelector('span') || btn;
    if (!btn.hasAttribute('data-original-text') && !isLoading) { // Store original only once
        btn.setAttribute('data-original-text', textSpan.textContent);
    }
    if (iconEl && !iconEl.hasAttribute('data-original-icon-class') && !isLoading && originalIcon) {
        iconEl.setAttribute('data-original-icon-class', `fas ${originalIcon} mr-2`);
    }

    if (isLoading) {
        textSpan.textContent = loadingText;
        if (iconEl) iconEl.className = 'fas fa-spinner fa-spin mr-2';
        else btn.classList.add('loading');
    } else {
        textSpan.textContent = btn.getAttribute('data-original-text') || textSpan.textContent;
        if (iconEl) iconEl.className = iconEl.getAttribute('data-original-icon-class') || (originalIcon ? `fas ${originalIcon} mr-2` : '');
        btn.classList.remove('loading');
    }
}

/**
 * 显示/隐藏主加载动画
 */
function showLoadingSpinner(show, message = '加载中...') {
    const spinner = DOM.loadingSpinnerContainer();
    if (!spinner) return;
    if (show) {
        spinner.querySelector('p').textContent = message;
        spinner.classList.remove('hidden');
        DOM.responseSection()?.classList.add('hidden');
    } else {
        spinner.classList.add('hidden');
    }
}

/**
 * 保存历史记录
 */
function saveToHistory(query, response, mode) {
    try {
        const history = JSON.parse(localStorage.getItem('queryHistory') || '[]');
        history.unshift({ query, answer: response.answer, mode, timestamp: new Date().toISOString() });
        if (history.length > 20) history.pop();
        localStorage.setItem('queryHistory', JSON.stringify(history));
    } catch (e) { console.warn("无法保存查询历史:", e); }
}

/**
 * 格式化回答 (使用marked.js)
 */
function formatAnswer(answerText) {
    if (!answerText) return '<p class="text-muted">AI未提供回答内容。</p>';
    if (typeof marked === 'undefined') {
        console.warn('marked.js 未加载，回退到基础格式化。');
        return `<p>${escapeHtml(answerText).replace(/\n/g, '<br>')}</p>`;
    }
    try {
        // For production, ensure output is sanitized if markdown comes from untrusted source
        // e.g. using DOMPurify: return DOMPurify.sanitize(marked.parse(answerText));
        return marked.parse(answerText);
    } catch (e) {
        console.error("Markdown解析错误:", e);
        return `<p class="text-error">无法解析回答内容。</p><pre>${escapeHtml(answerText)}</pre>`;
    }
}

/**
 * HTML转义
 */
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') unsafe = String(unsafe ?? '');
    return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

/**
 * 初始化示例问题
 */
function initExampleQueries() {
    const examples = [
        { text: "这份数据适合用什么模型进行分析？", query: "根据当前数据特点，推荐合适的分析模型。", mode: "data_analysis" },
        { text: "如果我想预测[目标列名]，哪些特征最重要？", query: "如果目标是[目标列名]，请分析各特征的重要性。", mode: "data_analysis" },
        { text: "使用[当前模型]模型对[目标列名]进行预测，并展示结果。", query: "请使用[当前模型]模型，以[目标列名]为目标进行预测，并告诉我预测结果和评估指标。", mode: "data_analysis" },
        { text: "请解释[当前模型]模型的原理及其优缺点。", query: "详细解释一下[当前模型]模型的工作原理、适用场景以及主要的优缺点是什么？", mode: "data_analysis" }, // Changed mode to general_llm as it's a knowledge question
        { text: "如何解读当前模型的[具体指标，如AUC或R方]？", query: "当前模型训练结果中的[具体指标，如AUC或R方]代表什么意义？这个数值表现如何？", mode: "data_analysis" },
        { text: "机器学习和传统编程有什么区别？", query: "机器学习与传统编程的主要区别是什么？", mode: "general_llm" },
        { text: "什么是监督学习和无监督学习？请举例说明。", query: "请解释监督学习和无监督学习的概念，并分别给出一个实际应用的例子。", mode: "general_llm" },
        { text: "在进行数据预处理时，通常需要注意哪些方面？", query: "数据预处理的关键步骤和常见注意事项有哪些？", mode: "general_llm" },
        { text: "如何避免模型训练中的过拟合问题？", query: "在模型训练过程中，有哪些常用的方法可以有效防止或减轻过拟合现象？", mode: "general_llm" },
        { text: "请介绍几种常见的模型评估指标及其适用场景。", query: "介绍几种常用的机器学习模型评估指标（例如准确率、精确率、召回率、F1分数、RMSE等），并说明它们各自适合在什么场景下使用。", mode: "general_llm" }
    ];
    const listEl = DOM.exampleQueryList();
    if (!listEl) return;
    listEl.innerHTML = '';
    examples.forEach(ex => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = "#"; a.textContent = ex.text;
        a.addEventListener('click', (e) => {
            e.preventDefault();
            // 替换模型名称并添加提示使用已有模型的文本
            let queryText = ex.text.replace("[选择的模型]", getModelDisplayName(selectedModelName) || "指定模型");
            // 如果是数据分析模式，确保查询明确表示使用已有模型
            if (ex.mode === "data_analysis" && !queryText.includes("已有") && !queryText.includes("现有")) {
                if (queryText.includes("预测")) {
                    queryText = queryText.replace("预测", "使用已有模型预测");
                }
            }
            DOM.queryInput().value = queryText;
            const radio = document.querySelector(`input[name="queryMode"][value="${ex.mode}"]`);
            if (radio) { radio.checked = true; radio.dispatchEvent(new Event('change')); }
            listEl.closest('.dropdown')?.removeAttribute('open');
            DOM.queryInput().focus();
        });
        li.appendChild(a); listEl.appendChild(li);
    });
}

/**
 * 初始化数据上传快捷按钮
 */
function initDataUploadShortcut() {
    DOM.uploadDataShortcutBtn()?.addEventListener('click', () => {
        document.getElementById('tab-link-dataUpload')?.click();
        if (DOM.uploadContainer()?.classList.contains('hidden')) DOM.toggleUploadBtn()?.click();
    });
}

/**
 * 显示Toast通知
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = DOM.toastContainer(); if (!container) return;
    const toastId = `toast-${Date.now()}`;
    const toast = document.createElement('div'); toast.id = toastId;
    const icons = { info: 'fa-info-circle', success: 'fa-check-circle', warning: 'fa-exclamation-triangle', error: 'fa-exclamation-circle' };
    toast.className = `alert alert-${type} shadow-lg mb-2 animate__animated animate__fadeInRight animate__faster`;
    toast.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${icons[type]} mr-3 text-xl"></i><span>${escapeHtml(message)}</span>
        </div>
        <button type="button" class="btn btn-sm btn-ghost btn-circle absolute top-1 right-1" onclick="hideToast('${toastId}')"><i class="fas fa-times"></i></button>`;
    container.appendChild(toast);
    if (toastTimeouts[toastId]) clearTimeout(toastTimeouts[toastId]);
    if (duration > 0) toastTimeouts[toastId] = setTimeout(() => hideToast(toastId), duration);
    return toastId;
}
window.hideToast = function(toastId) { // Expose to global for onclick
    const toast = document.getElementById(toastId); if (!toast) return;
    if (toastTimeouts[toastId]) { clearTimeout(toastTimeouts[toastId]); delete toastTimeouts[toastId]; }
    toast.classList.replace('animate__fadeInRight', 'animate__fadeOutRight');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
};

/**
 * 获取模型显示名称
 */
function getModelDisplayName(key) {
    if (!key) return '未知模型';
    const map = {
        'linear_regression': '线性回归',
        'random_forest_regressor': '随机森林回归',
        'logistic_regression': '逻辑回归',
        'decision_tree': '决策树',
        'random_forest_classifier': '随机森林分类',
        'knn_classifier': 'K近邻分类',
        'svm_classifier': '支持向量机(SVC)',
        'naive_bayes': '朴素贝叶斯',
        'kmeans': 'K均值聚类',
        'voting_classifier': '投票分类器',
        'voting_regressor': '投票回归器',
        'stacking_classifier': '堆叠分类器',
        'stacking_regressor': '堆叠回归器',
        'bagging_classifier': 'Bagging分类器',
        'bagging_regressor': 'Bagging回归器'
    };
    return map[key.toLowerCase()] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * 获取模型默认图标
 */
function getDefaultModelIcon(modelType) {
    if (!modelType) return 'fa-brain'; // Default icon
    const lowerType = modelType.toLowerCase();
    if (lowerType.includes('regression')) return 'fa-chart-line';
    if (lowerType.includes('classifier')) return 'fa-code-branch';
    if (lowerType.includes('kmeans') || lowerType.includes('clustering')) return 'fa-object-group';
    if (lowerType.includes('ensemble') || lowerType.includes('voting') || lowerType.includes('stacking') || lowerType.includes('bagging')) return 'fa-sitemap';
    if (lowerType.includes('tree')) return 'fa-tree';
    if (lowerType.includes('knn')) return 'fa-project-diagram';
    if (lowerType.includes('svm')) return 'fa-network-wired';
    if (lowerType.includes('naive_bayes')) return 'fa-percentage';
    return 'fa-brain'; // Fallback
}

/**
 * 初始化粒子背景
 */
function initParticlesJS() {
    const el = DOM.particlesBg();
    if (el && typeof particlesJS !== 'undefined') {
        setTimeout(() => {
            particlesJS('particles-js-bg', {
                particles: { number: { value: 60, density: { enable: true, value_area: 800 } }, color: { value: getComputedStyle(document.documentElement).getPropertyValue('--primary-hex').trim() }, shape: { type: "circle" }, opacity: { value: 0.3, random: true, anim: { enable: true, speed: 0.5, opacity_min: 0.05 } }, size: { value: 3, random: true }, line_linked: { enable: true, distance: 150, color: getComputedStyle(document.documentElement).getPropertyValue('--primary-hex').trim(), opacity: 0.2, width: 1 }, move: { enable: true, speed: 1, direction: "none", random: true, straight: false, out_mode: "out" } },
                interactivity: { detect_on: "canvas", events: { onhover: { enable: true, mode: "grab" }, onclick: { enable: false } }, modes: { grab: { distance: 140, line_linked: { opacity: 0.3 } } } },
                retina_detect: true
            });
        }, 200);
    }
}

// --- ADVANCED TOOLS ---
let allModelsCache = []; // Cache for all models to populate selectors

function initAdvancedTools() {
    initModelVersioning();
    initModelComparison();
    initEnsembleBuilding();
    initModelDeployment();
    
    // 加载默认数据集
    loadDefaultDatasets();

/**
 * 加载默认数据集
 */
function loadDefaultDatasets() {
    const dataSelect = DOM.compareTestDataSelect();
    if (dataSelect) {
        dataSelect.innerHTML = `
            <option value="" disabled selected>选择测试数据集</option>
            <option value="current_uploaded">当前上传数据</option>
        `;
    }
}
}

/**
 * Populate selectors for advanced tools (Model Versioning, Comparison, Ensemble)
 * This function is called AFTER loadAvailableModels in main, so allModelsCache should be ready.
 * If called independently, it will attempt to load models.
 * @param {Array<Object>} modelsList - Optional list of models to use, if already fetched.
 */
async function populateAdvancedToolSelectors(modelsList) {
    // If modelsList is provided, use it. Otherwise, assume allModelsCache is already populated by loadAvailableModels.
    if (modelsList && modelsList.length > 0) {
        allModelsCache = modelsList;
    } else if (allModelsCache.length === 0) { // If still empty, try fetching (fallback)
        try {
            const response = await fetch(API_ENDPOINTS.MODELS);
            const data = await response.json();
            allModelsCache = data.models || [];
        } catch (e) { console.error("Failed to fetch models for advanced tools:", e); }
    }

    populateModelSelector(DOM.versionModelSelector(), allModelsCache, "选择模型查看版本");
    populateModelSelector(DOM.deployModelSelect(), allModelsCache, "选择要部署的模型");
    // Dynamic selectors for compare/ensemble will be populated when added
}

function populateModelSelector(selectEl, models, placeholder) {
    if (!selectEl) return;
    const currentValue = selectEl.value; // Preserve selection if possible
    selectEl.innerHTML = `<option value="" disabled ${!currentValue ? 'selected' : ''}>${placeholder}</option>`;
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.internal_name || m.name;
        opt.textContent = m.name || getModelDisplayName(m.internal_name);
        if (opt.value === currentValue) opt.selected = true;
        selectEl.appendChild(opt);
    });
}

// --- MODEL VERSIONING ---
let currentModelForVersioning = null;
function initModelVersioning() {
    const selector = DOM.versionModelSelector();
    const createBtn = DOM.createVersionBtn();
    const saveBtn = DOM.saveVersionBtn();
    const cancelBtn = DOM.cancelSaveVersionBtn();
    if (!selector || !createBtn || !saveBtn || !cancelBtn) return;

    selector.addEventListener('change', async (e) => {
        currentModelForVersioning = e.target.value;
        DOM.versionMetadataForm().classList.add('hidden');
        if (currentModelForVersioning) await fetchAndDisplayModelVersions(currentModelForVersioning);
        else DOM.versionTableBody().innerHTML = `<tr><td colspan="5" class="text-center text-muted py-4">选择模型查看版本</td></tr>`;
    });
    createBtn.addEventListener('click', () => {
        if (!currentModelForVersioning) { showToast('请先选择模型。', 'warning'); return; }
        DOM.versionFormModelName().textContent = getModelDisplayName(currentModelForVersioning);
        DOM.versionDescription().value = ''; DOM.versionPerformance().value = '';
        DOM.versionMetadataForm().classList.remove('hidden'); DOM.versionDescription().focus();
    });
    cancelBtn.addEventListener('click', () => DOM.versionMetadataForm().classList.add('hidden'));
    saveBtn.addEventListener('click', async () => {
        if (!currentModelForVersioning) return;
        const desc = DOM.versionDescription().value.trim();
        const perf = DOM.versionPerformance().value.trim();
        if (!desc && !perf) { showToast('请输入版本描述或性能指标。', 'warning'); return; }
        setButtonLoading(saveBtn, true);
        try {
            // 调用创建模型版本API
            const response = await fetch(API_ENDPOINTS.MODEL_VERSIONS, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ 
                    model_name: currentModelForVersioning, 
                    description: desc, 
                    performance_metrics: perf 
                }) 
            });
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            
            showToast(result.message || `模型 "${getModelDisplayName(currentModelForVersioning)}" 新版本已创建。`, 'success');
            DOM.versionMetadataForm().classList.add('hidden');
            await fetchAndDisplayModelVersions(currentModelForVersioning);
        } catch (error) { showToast(`创建版本失败: ${error.message}`, 'error'); }
        finally { setButtonLoading(saveBtn, false); }
    });

}
async function fetchAndDisplayModelVersions(modelName) {
    const tBody = DOM.versionTableBody();
    tBody.innerHTML = `<tr><td colspan="5" class="text-center py-4"><span class="loading loading-dots"></span></td></tr>`;
    try {
        // 调用获取模型版本API
        const response = await fetch(`${API_ENDPOINTS.GET_MODEL_VERSIONS}${encodeURIComponent(modelName)}`);
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        
        const versions = result.versions || [];
        if (!versions.length) { 
            tBody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-4">无版本记录。</td></tr>`; 
            return; 
        }
        
        tBody.innerHTML = versions.map(v => `
            <tr class="hover"><td>${escapeHtml(v.id)}</td><td>${new Date(v.created_at).toLocaleString()}</td><td>${escapeHtml(v.description || '-')}</td><td>${escapeHtml(v.performance_metrics || '-')}</td>
            <td><button type="button" class="btn btn-xs btn-ghost tooltip" data-tip="回滚(未实现)" disabled><i class="fas fa-undo"></i></button></td></tr>`).join('');
    } catch (e) { 
        console.error('获取版本失败:', e);
        showToast(`获取版本失败: ${e.message}`, 'error'); 
        tBody.innerHTML = `<tr><td colspan="5" class="text-center text-error py-4">加载版本失败</td></tr>`; 
    }
}

// --- MODEL COMPARISON ---
let compareModelCount = 0; const MAX_COMPARE = 3;
function initModelComparison() {
    const addBtn = DOM.addCompareModelBtn();
    const startBtn = DOM.startCompareBtn();
    const dataSel = DOM.compareTestDataSelect();
    const targetSel = DOM.compareTargetColumnSelect();
    if (!addBtn || !startBtn) return;
    
    // 加载默认数据集
    loadDefaultDatasets();

/**
 * 加载默认数据集
 */
function loadDefaultDatasets() {
    const dataSelect = DOM.compareTestDataSelect();
    if (dataSelect) {
        dataSelect.innerHTML = `
            <option value="" disabled selected>选择测试数据集</option>
            <option value="current_uploaded">当前上传数据</option>
        `;
    }
}
    
    

    // 确保在初始化时已经加载了模型列表
    const initializeModelSelectors = async () => {
        // 如果模型缓存为空，先获取模型列表
        if (!allModelsCache || allModelsCache.length === 0) {
            try {
                const response = await fetch(API_ENDPOINTS.MODELS);
                const data = await response.json();
                allModelsCache = data.models || [];
            } catch (e) { 
                console.error("获取模型列表失败:", e); 
                showToast("获取模型列表失败，请刷新页面重试", "error");
            }
        }
        
        // 添加第一个模型选择器
        addDynamicModelSelector(
            DOM.compareModelsContainer(), 
            DOM.compareModelPlaceholder(), 
            allModelsCache, 
            compareModelCount, 
            MAX_COMPARE, 
            'compare-model-select', 
            '比较模型', 
            (nc) => compareModelCount = nc
        );
    };
    
    // 初始化模型选择器
    initializeModelSelectors();
    
    // 添加按钮事件监听
    addBtn.addEventListener('click', () => addDynamicModelSelector(
        DOM.compareModelsContainer(), 
        DOM.compareModelPlaceholder(), 
        allModelsCache,
        compareModelCount, 
        MAX_COMPARE, 
        'compare-model-select', 
        '比较模型',
        (newCount) => compareModelCount = newCount
    ));

    startBtn.addEventListener('click', async () => {
        const models = Array.from(DOM.compareModelsContainer().querySelectorAll('.compare-model-select')).map(s => s.value).filter(Boolean);
        const testData = dataSel.value; const target = targetSel.value;
        if (models.length < 2) { showToast('请至少选择两个模型。', 'warning'); return; }
        if (!testData) { showToast('请选择测试数据集。', 'warning'); return; }
        if (testData === 'current_uploaded' && (!currentData.path || !currentData.analysisCompleted || !target)) { showToast('当前数据未就绪或未选目标列。', 'warning'); return; }
        DOM.compareResultsContainer().innerHTML = `<p class="text-center py-4"><span class="loading loading-lg loading-dots"></span></p>`;
        setButtonLoading(startBtn, true);
        try {
            // 调用模型比较API
            const response = await fetch(API_ENDPOINTS.COMPARE_MODELS, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ 
                    model_names: models, 
                    test_data_path: testData, 
                    target_column: target 
                }) 
            });
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            
            // 显示比较结果
            let html = `<div class="prose max-w-none"><h3>比较结果</h3><table class="table table-zebra w-full"><thead><tr><th>模型</th>`;
            
            // 确定所有可能的指标
            const allMetrics = new Set();
            result.comparison_results.forEach(model => {
                if (model.metrics) {
                    Object.keys(model.metrics).forEach(metric => allMetrics.add(metric));
                }
            });
            
            // 添加指标列
            allMetrics.forEach(metric => {
                html += `<th>${formatMetricName(metric)}</th>`;
            });
            
            html += `</tr></thead><tbody>`;
            
            // 添加每个模型的结果行
            result.comparison_results.forEach((model, index) => {
                html += `<tr class="${index % 2 === 0 ? 'bg-base-200' : ''}"><td>${model.model_name}</td>`;
                
                allMetrics.forEach(metric => {
                    const value = model.metrics && model.metrics[metric] !== undefined ? 
                        model.metrics[metric].toFixed(4) : '-';
                    html += `<td>${value}</td>`;
                });
                
                html += `</tr>`;
            });
            
            html += `</tbody></table>`;
            
            // 添加测试数据信息
            if (result.test_data) {
                html += `<div class="mt-4">
                    <h4>测试数据信息</h4>
                    <p>路径: ${result.test_data.path}</p>
                    <p>行数: ${result.test_data.rows}</p>
                    <p>列数: ${result.test_data.columns}</p>
                </div>`;
            }
            
            html += `</div>`;
            
            DOM.compareResultsContainer().innerHTML = html;
        } catch (e) { 
            console.error('比较模型错误:', e);
            showToast(`比较失败: ${e.message}`, 'error'); 
            DOM.compareResultsContainer().innerHTML = `<p class="text-error text-center">比较失败: ${e.message}</p>`; 
        }
        finally { setButtonLoading(startBtn, false); }
    });

    dataSel.addEventListener('change', async () => {
        // 当选择测试数据集时，加载相应的目标列
        if (!dataSel.value) return;
        
        targetSel.innerHTML = '<option value="" disabled selected>加载中...</option>';
        
        if (dataSel.value === 'current_uploaded') {
            // 使用当前上传的数据集的列
            populateSelectWithOptions(targetSel, currentData.columns, "选择目标列");
        } else {
            // 从服务器获取数据集的列
            try {
                const response = await fetch(`/api/ml/analyze?file_path=${encodeURIComponent(dataSel.value)}`);
                const result = await response.json();
                if (result.error) throw new Error(result.error);
                
                const columns = result.columns || [];
                populateSelectWithOptions(targetSel, columns, "选择目标列");
            } catch (e) {
                console.error('获取数据集列失败:', e);
                populateSelectWithOptions(targetSel, [], "获取列失败");
                showToast(`获取数据集列失败: ${e.message}`, 'error');
            }
        }
    });

/**
 * 用选项填充选择器
 */
function populateSelectWithOptions(selectElement, options, placeholderText = "请选择") {
    if (!selectElement) return;
    
    // 清除现有选项
    selectElement.innerHTML = '';
    
    // 添加占位符选项
    const placeholderOption = document.createElement('option');
    placeholderOption.value = '';
    placeholderOption.textContent = placeholderText;
    placeholderOption.disabled = true;
    placeholderOption.selected = true;
    selectElement.appendChild(placeholderOption);
    
    // 添加所有选项
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        selectElement.appendChild(opt);
    });
}
}

/**
 * 格式化指标名称
 */
function formatMetricName(metric) {
    const metricMap = {
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1_score': 'F1分数',
        'r2_score': 'R²分数',
        'mean_squared_error': '均方误差',
        'root_mean_squared_error': '均方根误差'
    };
    
    return metricMap[metric] || metric;
}

// --- ENSEMBLE BUILDING ---
let ensembleModelCount = 0; const MIN_ENSEMBLE = 2;
function initEnsembleBuilding() {
    const addBtn = DOM.addEnsembleModelBtn();
    const buildBtn = DOM.buildEnsembleBtn();
    if (!addBtn || !buildBtn) return;
    
    // 确保模型列表已加载
    populateAdvancedToolSelectors();
    
    // 加载所有可用模型并初始化选择器
    setTimeout(async () => {
        if (!allModelsCache || allModelsCache.length === 0) {
            try {
                const response = await fetch(API_ENDPOINTS.MODELS);
                if (!response.ok) throw new Error(`请求失败 (${response.status})`);
                const data = await response.json();
                allModelsCache = data.models || [];
                if (allModelsCache.length === 0) {
                    console.warn('没有可用的模型');
                    return;
                }
            } catch (error) {
                console.error('获取模型列表失败:', error);
                return;
            }
        }
        
        // 初始化选择器值
        const selectors = document.querySelectorAll('.ensemble-model-select');
        selectors.forEach((selector, index) => {
            if (index < allModelsCache.length) {
                const model = allModelsCache[index];
                selector.value = model.internal_name || model.name;
                selector.title = model.description || model.display_name || model.name;
            }
        });
    }, 500);

    addBtn.addEventListener('click', () => {
        if (!allModelsCache || allModelsCache.length === 0) {
            loadAvailableModels().then(() => {
                addDynamicModelSelector(
                    DOM.ensembleModelsContainer(), 
                    DOM.ensembleModelPlaceholder(), 
                    allModelsCache.filter(m => m.type !== 'ensemble'),
                    ensembleModelCount, 
                    10, 
                    'ensemble-model-select', 
                    '基础模型',
                    (newCount) => ensembleModelCount = newCount
                );
            });
        } else {
            addDynamicModelSelector(
                DOM.ensembleModelsContainer(), 
                DOM.ensembleModelPlaceholder(), 
                allModelsCache.filter(m => m.type !== 'ensemble'),
                ensembleModelCount, 
                10, 
                'ensemble-model-select', 
                    '基础模型',
                    (newCount) => ensembleModelCount = newCount
            );
        }
    });
    // 初始化添加两个基础模型选择器
    for(let i=0; i<MIN_ENSEMBLE; i++) addDynamicModelSelector(DOM.ensembleModelsContainer(), DOM.ensembleModelPlaceholder(), allModelsCache.filter(m => m.type !== 'ensemble'), ensembleModelCount, 10, 'ensemble-model-select', '基础模型', (nc) => ensembleModelCount = nc);

    buildBtn.addEventListener('click', async () => {
        const models = Array.from(DOM.ensembleModelsContainer().querySelectorAll('.ensemble-model-select')).map(s => s.value).filter(Boolean);
        const type = DOM.ensembleTypeSelect().value;
        const name = DOM.ensembleName().value.trim();
        
        // 验证输入
        if (models.length < MIN_ENSEMBLE) { 
            showToast(`请至少选择 ${MIN_ENSEMBLE} 个基础模型。`, 'warning'); 
            return; 
        }
        if (!type) {
            showToast('请选择集成类型。', 'warning');
            return;
        }
        if (!name) {
            showToast('请输入集成模型名称。', 'warning');
            return;
        }
        if (!/^[a-zA-Z0-9_.-]+$/.test(name)) { 
            showToast('模型名称只能包含字母、数字、下划线、点和连字符。', 'warning'); 
            return; 
        }
        
        // 显示加载状态
        DOM.ensembleResultContainer().innerHTML = `<p class="text-center py-4"><span class="loading loading-lg loading-dots"></span></p>`;
        setButtonLoading(buildBtn, true, '构建中...');
        
        try {
            // 调用构建集成模型API
            const response = await fetch(API_ENDPOINTS.BUILD_ENSEMBLE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_models: models,
                    ensemble_type: type,
                    save_name: name
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `请求失败 (${response.status})`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || '未知错误');
            }
            
            // 显示成功结果
            DOM.ensembleResultContainer().innerHTML = `
                <div class="prose max-w-none">
                    <p class="text-success font-medium">集成模型 "${escapeHtml(name)}" 构建成功！</p>
                    <div class="mt-3 p-3 bg-base-200 rounded-lg">
                        <p class="text-sm mb-2"><span class="font-medium">模型类型:</span> ${result.ensemble_type} 集成</p>
                        <p class="text-sm mb-2"><span class="font-medium">基础模型:</span> ${result.base_models.map(m => getModelDisplayName(m)).join(', ')}</p>
                        <p class="text-sm"><span class="font-medium">创建时间:</span> ${new Date(result.model_info.created_at).toLocaleString()}</p>
                    </div>
                </div>`;
                
            showToast('集成模型构建成功！', 'success');
            await loadAvailableModels(); // 刷新所有模型列表
        } catch (e) { 
            console.error('构建集成模型错误:', e);
            showToast(`构建失败: ${e.message}`, 'error'); 
            DOM.ensembleResultContainer().innerHTML = `<p class="text-error text-center py-4">构建失败: ${escapeHtml(e.message)}</p>`; 
        } finally { 
            setButtonLoading(buildBtn, false, '<i class="fas fa-magic" aria-hidden="true"></i> 构建集成模型'); 
        }
    });
}

/** Generic function to add a model selector dynamically */
function addDynamicModelSelector(container, placeholderEl, modelsList, currentCount, maxCount, selectClass, labelPrefix, updateCountCallback) {
    if (currentCount >= maxCount) {
        showToast(`最多只能选择 ${maxCount} 个${labelPrefix}。`, 'warning'); return;
    }
    
    // For ensemble, ensure at least MIN_ENSEMBLE remain before allowing removal of the first one
    const allowRemoval = labelPrefix !== '基础模型' || currentCount >= MIN_ENSEMBLE;

    currentCount++;
    if (placeholderEl && currentCount === 1) placeholderEl.classList.add('hidden');
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2 mb-2 dynamic-selector-row group'; // Added group for easier styling
    div.innerHTML = `<label for="${selectClass}${currentCount}" class="sr-only">${labelPrefix} ${currentCount}</label>
        <select id="${selectClass}${currentCount}" class="${selectClass} select select-sm select-bordered w-full"></select>
        <button type="button" class="btn btn-xs btn-circle btn-ghost text-error remove-dynamic-selector-btn ${allowRemoval ? '' : 'invisible group-hover:visible'}" aria-label="移除" ${allowRemoval ? '' : 'disabled'}><i class="fas fa-times"></i></button>`;
    populateModelSelector(div.querySelector('select'), modelsList, `${labelPrefix} ${currentCount}`);
    div.querySelector('.remove-dynamic-selector-btn')?.addEventListener('click', () => {
        div.remove();
        currentCount--;
        if (currentCount === 0 && placeholderEl) placeholderEl.classList.remove('hidden');
        updateCountCallback(currentCount);
        // Re-evaluate remove button visibility after removal
        container.querySelectorAll('.dynamic-selector-row').forEach((row, index, list) => {
             const removeBtn = row.querySelector('.remove-dynamic-selector-btn');
             // Only hide/disable the remove button if it's an ensemble row and we are at the minimum count
             const isEnsemble = selectClass === 'ensemble-model-select';
             const isBelowMin = list.length < MIN_ENSEMBLE;
             if (isEnsemble && isBelowMin) {
                  removeBtn?.classList.add('invisible');
                  removeBtn?.setAttribute('disabled', 'true');
             } else {
                 removeBtn?.classList.remove('invisible');
                  removeBtn?.removeAttribute('disabled');
             }
        });
    });
    container.appendChild(div);
    updateCountCallback(currentCount);
}


// --- MODEL DEPLOYMENT & MONITORING ---
function initModelDeployment() {
    const deployBtn = DOM.deployModelBtn();
    const refreshBtn = DOM.refreshMonitorBtn();
    if (!deployBtn || !refreshBtn) return;

    deployBtn.addEventListener('click', async () => {
        const modelName = DOM.deployModelSelect().value;
        const environment = DOM.deployEnvironmentSelect().value;
        const endpointPathSuggestion = DOM.deployEndpoint().value.trim(); // 端点路径建议，后端会生成最终路径
        
        // 验证输入
        if (!modelName) {
            showToast('请选择要部署的模型。', 'warning');
            return;
        }
        if (!environment) {
            showToast('请选择部署环境。', 'warning');
            return;
        }
        if (!endpointPathSuggestion) {
            showToast('请输入API端点名称。', 'warning');
            return;
        }
        if (!/^\/[a-zA-Z0-9\/_-]+$/.test(endpointPathSuggestion)) { 
            showToast('端点路径格式无效，应以斜杠开头，只包含字母、数字、下划线和连字符 (例如 /predict/my_model)。', 'warning'); 
            return; 
        }
        
        // 显示加载状态
        setButtonLoading(deployBtn, true, '部署中...');
        
        try {
            const payload = { 
                model_name: modelName, 
                environment: environment
                // 后端会根据模型名称和环境生成端点路径
            };

            const response = await fetch(API_ENDPOINTS.DEPLOY_MODEL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `请求失败 (${response.status})`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || '未知错误');
            }
            
            // 显示成功消息
            showToast(result.message || `模型 "${getModelDisplayName(modelName)}" 已成功部署到 ${result.environment} 环境。`, 'success', 7000);
            
            // 更新端点输入框
            DOM.deployEndpoint().value = result.endpoint_url || '';
            
            // 刷新部署列表
            await fetchAndDisplayDeployedModels();
        } catch (e) { 
            console.error('部署模型错误:', e);
            showToast(`部署失败: ${e.message}`, 'error');
        } finally { 
            setButtonLoading(deployBtn, false, '<i class="fas fa-cloud-upload-alt" aria-hidden="true"></i> 部署模型'); 
        }
    });
    refreshBtn.addEventListener('click', fetchAndDisplayDeployedModels);
    fetchAndDisplayDeployedModels(); // Initial load
}
async function fetchAndDisplayDeployedModels() {
    const tBody = DOM.deploymentTableBody();
    const refreshBtn = DOM.refreshMonitorBtn();
    if (refreshBtn) setButtonLoading(refreshBtn, true, '刷新中...', refreshBtn.querySelector('i'), 'fa-sync-alt');
    tBody.innerHTML = `<tr><td colspan="5" class="text-center py-4"><span class="loading loading-dots"></span></td></tr>`;
    
    try {
        // 调用获取已部署模型API
        const response = await fetch(API_ENDPOINTS.DEPLOYED_MODELS);
        
        if (!response.ok) {
            throw new Error(`请求失败 (${response.status})`);
        }
        
        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || '获取部署列表失败');
        }
        
        const deployments = result.deployments || [];
        
        if (!deployments.length) { 
            tBody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-4">暂无已部署模型。</td></tr>`; 
        } else {
            tBody.innerHTML = deployments.map(d => `
                <tr class="hover">
                    <td>${getModelDisplayName(d.model_name)}</td>
                    <td><span class="badge badge-sm ${d.environment === 'production' ? 'badge-error' : d.environment === 'staging' ? 'badge-warning' : 'badge-info'}">${d.environment}</span></td>
                    <td><code>${d.endpoint}</code></td>
                    <td><span class="badge badge-sm ${d.status === '运行中' ? 'badge-success' : 'badge-ghost'}">${d.status}</span></td>
                    <td><button type="button" class="btn btn-xs btn-ghost text-error" onclick="confirmUndeploy('${d.id}','${d.model_name}')" ${d.status !== '运行中' ? 'disabled':''}><i class="fas fa-stop-circle"></i></button></td>
                </tr>`).join('');
        }
        
        // 更新统计信息
        DOM.deployedModelCount().textContent = result.count || 0;
        DOM.avgResponseTime().textContent = result.avg_response_time ? `${Math.round(result.avg_response_time)}ms` : 'N/A';
        DOM.predictionRequests().textContent = result.total_requests || 0;
    } catch (e) { 
        console.error('获取部署列表错误:', e);
        showToast(`获取部署列表失败: ${e.message}`, 'error'); 
        tBody.innerHTML = `<tr><td colspan="5" class="text-center text-error py-4">刷新监控失败: ${escapeHtml(e.message)}</td></tr>`; 
    } finally { 
        if (refreshBtn) setButtonLoading(refreshBtn, false, '刷新', refreshBtn.querySelector('i'), 'fa-sync-alt'); 
    }
}
window.confirmUndeploy = async function(id, name) { // 暴露给onclick使用
    // 使用确认对话框
    if (!confirm(`确定要停止部署模型 "${name}" (ID: ${id})?`)) return;
    
    showToast(`正在请求停止部署模型 "${name}"...`, 'info');
    try {
        const response = await fetch(`${API_ENDPOINTS.UNDEPLOY_MODEL}${encodeURIComponent(id)}`, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
            // 不需要请求体，后端通过路径参数获取部署ID
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `请求失败 (${response.status})`);
        }
        
        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || '未知错误');
        }
        
        showToast(result.message || `模型 "${name}" 已成功停止部署。`, 'success');
        await fetchAndDisplayDeployedModels(); // 刷新列表
    } catch (e) { 
        console.error('停止部署失败:', e);
        showToast(`停止部署模型 "${name}" 失败: ${e.message}`, 'error');
    }
};

// DOM加载完成后执行
document.addEventListener('DOMContentLoaded', main);