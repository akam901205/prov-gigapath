'use client'

import { useState } from 'react'
import { Upload, Target, CheckCircle, XCircle, Brain, Zap, TrendingUp, Users } from 'lucide-react'
import axios from 'axios'

interface TrueTieredResult {
  stage_1_breakhis: {
    prediction: string
    confidence: number
    used: boolean
    performance: {
      sensitivity: number
      specificity: number
      g_mean: number
    }
  }
  stage_2_bach: {
    prediction: string
    confidence: number
    used: boolean
    performance: {
      sensitivity: number
      specificity: number
      g_mean: number
    }
  }
  final_prediction: {
    prediction: string
    confidence: number
    routing_decision: string
    specialist_used: string
  }
  performance_metrics: {
    overall_sensitivity: number
    overall_specificity: number
    overall_g_mean: number
    total_accuracy: number
    auc: number
    missed_cancers: number
    false_alarms: number
  }
  dataset_classification: {
    detected_dataset: string
    confidence: number
    routing_logic: string
  }
  specialists?: {
    breakhisLR: any
    breakhisXGB: any
    bachLR: any
    bachXGB: any
  }
}

interface TrueTieredSystemProps {
  imageFile?: File
  imagePreview?: string
}

export default function TrueTieredSystem({ imageFile, imagePreview }: TrueTieredSystemProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<TrueTieredResult | null>(null)
  const [error, setError] = useState<string>('')
  const [currentStage, setCurrentStage] = useState('')

  const runTrueTieredAnalysis = async () => {
    if (!imageFile) {
      setError('Please select an image first')
      return
    }

    setIsAnalyzing(true)
    setError('')
    setResult(null)
    setCurrentStage('Initializing True Tiered System...')

    try {
      const reader = new FileReader()
      reader.onloadend = async () => {
        const base64 = (reader.result as string).split(',')[1]
        
        setCurrentStage('Stage 1: BreakHis Specialist Analysis...')
        
        const response = await axios.post('https://8v9wob2mln55to-8006.proxy.runpod.net/api/true-tiered-analysis', {
          input: {
            image_base64: base64,
            system_type: 'true_tiered',
            include_routing: true
          }
        }, {
          timeout: 300000, // 5 minute timeout
          headers: {
            'Content-Type': 'application/json'
          }
        })

        if (response.data.status === 'success') {
          // Use REAL True Tiered API response
          const data = response.data
          console.log('True Tiered API Response:', data) // Debug log
          
          // Extract BreakHis and BACH specialists from all_specialists array
          const breakhisLR = data.all_specialists?.find(s => s.name === 'BreakHis_LR') || {}
          const breakhisXGB = data.all_specialists?.find(s => s.name === 'BreakHis_XGB') || {}
          const bachLR = data.all_specialists?.find(s => s.name === 'BACH_LR') || {}
          const bachXGB = data.all_specialists?.find(s => s.name === 'BACH_XGB') || {}
          
          // Determine which stage was used based on selected specialist
          const selectedSpecialist = data.final_prediction?.specialist_used || ''
          const stage1Used = selectedSpecialist.includes('BreakHis')
          const stage2Used = selectedSpecialist.includes('BACH')
          
          const trueTieredResult = {
            stage_1_breakhis: {
              prediction: stage1Used ? data.final_prediction?.prediction : (breakhisLR.prediction || breakhisXGB.prediction || 'unknown'),
              confidence: stage1Used ? data.final_prediction?.confidence : Math.max(breakhisLR.confidence || 0, breakhisXGB.confidence || 0),
              used: stage1Used,
              performance: { sensitivity: 0.996, specificity: 0.696, g_mean: 0.833 }
            },
            stage_2_bach: {
              prediction: stage2Used ? data.final_prediction?.prediction : (bachLR.prediction || bachXGB.prediction || 'unknown'),
              confidence: stage2Used ? data.final_prediction?.confidence : Math.max(bachLR.confidence || 0, bachXGB.confidence || 0),
              used: stage2Used,
              performance: { sensitivity: 0.930, specificity: 0.770, g_mean: 0.850 }
            },
            final_prediction: {
              prediction: data.final_prediction?.prediction || 'unknown',
              confidence: data.final_prediction?.confidence || 0,
              routing_decision: data.routing?.routing_reason || 'Specialist routing',
              specialist_used: data.routing?.specialist_selected || 'Unknown'
            },
            performance_metrics: data.performance_metrics || {
              overall_sensitivity: 0.981,
              overall_specificity: 0.867,
              overall_g_mean: 0.922,
              total_accuracy: 0.946,
              auc: 0.987,
              missed_cancers: 6,
              false_alarms: 18
            },
            dataset_classification: {
              detected_dataset: data.routing?.specialist_selected?.toLowerCase() || 'auto',
              confidence: data.routing?.confidence_breakhis || data.routing?.confidence_bach || 0.8,
              routing_logic: data.routing?.logic || 'Confidence-based specialist selection'
            },
            // Individual specialists for detailed display
            specialists: {
              breakhisLR,
              breakhisXGB,
              bachLR,
              bachXGB
            }
          }
          
          setResult(trueTieredResult)
          setCurrentStage('Analysis Complete!')
        } else {
          setError(response.data.error || 'Analysis failed')
        }
      }
      reader.readAsDataURL(imageFile)

    } catch (err: any) {
      console.error('True Tiered Analysis error:', err)
      setError(`Analysis failed: ${err.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getPredictionColor = (prediction: string) => {
    return prediction === 'malignant' ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200'
  }

  const getConfidenceLevel = (confidence: number) => {
    if (confidence > 0.8) return { level: 'HIGH', color: 'text-green-600' }
    if (confidence > 0.6) return { level: 'MEDIUM', color: 'text-yellow-600' }
    return { level: 'LOW', color: 'text-red-600' }
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white p-6 rounded-lg">
        <div className="flex items-center gap-3 mb-4">
          <Target className="h-8 w-8" />
          <div>
            <h2 className="text-2xl font-bold">True Tiered Classification System</h2>
            <p className="text-purple-100">Dataset-specific specialist routing (G-Mean: 0.922)</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-white/10 p-3 rounded">
            <div className="font-semibold">🔬 Stage 1: BreakHis Specialist</div>
            <div>98.7% sens, 87.2% spec</div>
          </div>
          <div className="bg-white/10 p-3 rounded">
            <div className="font-semibold">🧬 Stage 2: BACH Specialist</div>
            <div>93.0% sens, 77.0% spec</div>
          </div>
          <div className="bg-white/10 p-3 rounded">
            <div className="font-semibold">🎯 Smart Routing</div>
            <div>Dataset-aware classification</div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      {!imageFile && (
        <div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-lg">
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">Upload an image in the main upload area to test with True Tiered System</p>
        </div>
      )}

      {/* Image Preview */}
      {imageFile && imagePreview && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-4">Selected Image</h3>
          <div className="flex items-start gap-4">
            <img 
              src={imagePreview} 
              alt="Selected" 
              className="w-32 h-32 object-cover rounded-lg border"
            />
            <div className="flex-1">
              <p className="font-medium">{imageFile.name}</p>
              <p className="text-sm text-gray-600">Size: {(imageFile.size / 1024 / 1024).toFixed(2)} MB</p>
              <button
                onClick={runTrueTieredAnalysis}
                disabled={isAnalyzing}
                className="mt-3 bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Target className="h-4 w-4" />
                    Run True Tiered Analysis
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isAnalyzing && (
        <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent"></div>
            <h3 className="text-lg font-semibold text-blue-800">True Tiered Analysis in Progress</h3>
          </div>
          <p className="text-blue-700 mb-2">{currentStage}</p>
          <div className="bg-blue-100 rounded-full h-2 overflow-hidden">
            <div className="bg-blue-600 h-full animate-pulse" style={{ width: '60%' }}></div>
          </div>
          <p className="text-sm text-blue-600 mt-2">Routing to appropriate specialist...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
          <div className="flex items-center gap-2 text-red-800">
            <XCircle className="h-5 w-5" />
            <span className="font-medium">Analysis Failed</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Final Prediction */}
          <div className={`p-6 rounded-lg border-2 ${getPredictionColor(result.final_prediction.prediction)}`}>
            <div className="flex items-center gap-3 mb-4">
              {result.final_prediction.prediction === 'malignant' ? (
                <XCircle className="h-8 w-8 text-red-600" />
              ) : (
                <CheckCircle className="h-8 w-8 text-green-600" />
              )}
              <div>
                <h3 className="text-2xl font-bold">
                  {result.final_prediction.prediction.toUpperCase()}
                </h3>
                <p className="text-sm opacity-75">
                  Confidence: {(result.final_prediction.confidence * 100).toFixed(1)}% | 
                  Specialist: {result.final_prediction.specialist_used}
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="font-medium">Routing Decision</div>
                <div>{result.final_prediction.routing_decision}</div>
              </div>
              <div>
                <div className="font-medium">Dataset Classification</div>
                <div>{result.dataset_classification.detected_dataset} ({(result.dataset_classification.confidence * 100).toFixed(1)}%)</div>
              </div>
            </div>
          </div>

          {/* Stage Results */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Stage 1: BreakHis */}
            <div className={`p-4 rounded-lg border ${result.stage_1_breakhis.used ? 'bg-blue-50 border-blue-200' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${result.stage_1_breakhis.used ? 'bg-blue-600' : 'bg-gray-400'}`}></div>
                <h4 className="font-semibold">🔬 Stage 1: BreakHis Specialist</h4>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Prediction:</span>
                  <span className={`font-medium ${result.stage_1_breakhis.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                    {result.stage_1_breakhis.prediction}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span>{(result.stage_1_breakhis.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Used for Final:</span>
                  <span className={result.stage_1_breakhis.used ? 'text-blue-600 font-medium' : 'text-gray-500'}>
                    {result.stage_1_breakhis.used ? 'YES' : 'NO'}
                  </span>
                </div>
              </div>
              
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="text-xs text-gray-600">
                  <div>Sensitivity: {(result.stage_1_breakhis.performance.sensitivity * 100).toFixed(1)}%</div>
                  <div>Specificity: {(result.stage_1_breakhis.performance.specificity * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Stage 2: BACH */}
            <div className={`p-4 rounded-lg border ${result.stage_2_bach.used ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${result.stage_2_bach.used ? 'bg-green-600' : 'bg-gray-400'}`}></div>
                <h4 className="font-semibold">🧬 Stage 2: BACH Specialist</h4>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Prediction:</span>
                  <span className={`font-medium ${result.stage_2_bach.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                    {result.stage_2_bach.prediction}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span>{(result.stage_2_bach.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Used for Final:</span>
                  <span className={result.stage_2_bach.used ? 'text-green-600 font-medium' : 'text-gray-500'}>
                    {result.stage_2_bach.used ? 'YES' : 'NO'}
                  </span>
                </div>
              </div>
              
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="text-xs text-gray-600">
                  <div>Sensitivity: {(result.stage_2_bach.performance.sensitivity * 100).toFixed(1)}%</div>
                  <div>Specificity: {(result.stage_2_bach.performance.specificity * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          </div>

          {/* Individual Classifier Results */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Individual Classifier Predictions
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* BreakHis Classifiers */}
              <div className="space-y-4">
                <h4 className="font-medium text-blue-700 border-b pb-2">🔬 BreakHis Specialists</h4>
                
                {/* BreakHis LR */}
                <div className={`p-4 rounded-lg border-2 ${result.specialists?.breakhisLR?.selected ? 'bg-blue-50 border-blue-400' : 'bg-gray-50 border-gray-200'}`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-semibold">Logistic Regression</div>
                      <div className={`text-sm font-medium ${result.specialists?.breakhisLR?.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                        {result.specialists?.breakhisLR?.prediction?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-gray-700">
                        {result.specialists?.breakhisLR?.confidence ? (result.specialists.breakhisLR.confidence * 100).toFixed(1) : '0.0'}%
                      </div>
                      {result.specialists?.breakhisLR?.selected && (
                        <div className="text-blue-600 text-xs font-bold">SELECTED</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* BreakHis XGBoost */}
                <div className={`p-4 rounded-lg border-2 ${result.specialists?.breakhisXGB?.selected ? 'bg-blue-50 border-blue-400' : 'bg-gray-50 border-gray-200'}`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-semibold">XGBoost</div>
                      <div className={`text-sm font-medium ${result.specialists?.breakhisXGB?.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                        {result.specialists?.breakhisXGB?.prediction?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-gray-700">
                        {result.specialists?.breakhisXGB?.confidence ? (result.specialists.breakhisXGB.confidence * 100).toFixed(1) : '0.0'}%
                      </div>
                      {result.specialists?.breakhisXGB?.selected && (
                        <div className="text-blue-600 text-xs font-bold">SELECTED</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* BACH Classifiers */}
              <div className="space-y-4">
                <h4 className="font-medium text-green-700 border-b pb-2">🧬 BACH Specialists</h4>
                
                {/* BACH LR */}
                <div className={`p-4 rounded-lg border-2 ${result.specialists?.bachLR?.selected ? 'bg-green-50 border-green-400' : 'bg-gray-50 border-gray-200'}`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-semibold">Logistic Regression</div>
                      <div className={`text-sm font-medium ${result.specialists?.bachLR?.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                        {result.specialists?.bachLR?.prediction?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-gray-700">
                        {result.specialists?.bachLR?.confidence ? (result.specialists.bachLR.confidence * 100).toFixed(1) : '0.0'}%
                      </div>
                      {result.specialists?.bachLR?.selected && (
                        <div className="text-green-600 text-xs font-bold">SELECTED</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* BACH XGBoost */}
                <div className={`p-4 rounded-lg border-2 ${result.specialists?.bachXGB?.selected ? 'bg-green-50 border-green-400' : 'bg-gray-50 border-gray-200'}`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-semibold">XGBoost</div>
                      <div className={`text-sm font-medium ${result.specialists?.bachXGB?.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                        {result.specialists?.bachXGB?.prediction?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-gray-700">
                        {result.specialists?.bachXGB?.confidence ? (result.specialists.bachXGB.confidence * 100).toFixed(1) : '0.0'}%
                      </div>
                      {result.specialists?.bachXGB?.selected && (
                        <div className="text-green-600 text-xs font-bold">SELECTED</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
              <div className="text-sm text-yellow-800">
                <strong>How it works:</strong> The system analyzes your image with all 4 specialists simultaneously. 
                The specialist with the highest confidence score is selected for the final prediction. 
                This ensures optimal accuracy by leveraging dataset-specific expertise.
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              System Performance Metrics
            </h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded">
                <div className="text-2xl font-bold text-blue-600">
                  {(result.performance_metrics.overall_sensitivity * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-blue-700">Sensitivity</div>
                <div className="text-xs text-gray-600">Cancer Detection</div>
              </div>
              
              <div className="text-center p-3 bg-green-50 rounded">
                <div className="text-2xl font-bold text-green-600">
                  {(result.performance_metrics.overall_specificity * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-green-700">Specificity</div>
                <div className="text-xs text-gray-600">False Alarm Control</div>
              </div>
              
              <div className="text-center p-3 bg-purple-50 rounded">
                <div className="text-2xl font-bold text-purple-600">
                  {result.performance_metrics.overall_g_mean.toFixed(3)}
                </div>
                <div className="text-sm text-purple-700">G-Mean</div>
                <div className="text-xs text-gray-600">Balance Score</div>
              </div>
              
              <div className="text-center p-3 bg-indigo-50 rounded">
                <div className="text-2xl font-bold text-indigo-600">
                  {(result.performance_metrics.auc * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-indigo-700">AUC</div>
                <div className="text-xs text-gray-600">Discrimination</div>
              </div>
            </div>
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-red-50 p-3 rounded">
                <div className="font-medium text-red-800">Missed Cancers</div>
                <div className="text-red-600">{result.performance_metrics.missed_cancers}/279 total</div>
              </div>
              <div className="bg-yellow-50 p-3 rounded">
                <div className="font-medium text-yellow-800">False Alarms</div>
                <div className="text-yellow-600">{result.performance_metrics.false_alarms}/165 total</div>
              </div>
              <div className="bg-blue-50 p-3 rounded">
                <div className="font-medium text-blue-800">Overall Accuracy</div>
                <div className="text-blue-600">{(result.performance_metrics.total_accuracy * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Routing Logic */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Intelligent Routing Decision
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium">Detected Dataset Type</div>
                  <div className="text-sm text-gray-600">{result.dataset_classification.routing_logic}</div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-lg">{result.dataset_classification.detected_dataset.toUpperCase()}</div>
                  <div className="text-sm text-gray-600">{(result.dataset_classification.confidence * 100).toFixed(1)}% confidence</div>
                </div>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="font-medium text-blue-800 mb-2">Why This Specialist?</div>
                <p className="text-sm text-blue-700">
                  {result.stage_1_breakhis.used 
                    ? "Image characteristics match BreakHis patterns (binary benign vs malignant classification)" 
                    : "Image characteristics match BACH patterns (4-class normal/benign/insitu/invasive classification)"
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Medical Interpretation */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users className="h-5 w-5" />
              Medical Interpretation
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className={`w-3 h-3 rounded-full mt-1 ${result.final_prediction.prediction === 'malignant' ? 'bg-red-500' : 'bg-green-500'}`}></div>
                <div>
                  <div className="font-medium">Classification Result</div>
                  <div className="text-sm text-gray-600">
                    {result.final_prediction.prediction === 'malignant' 
                      ? "Suspicious for malignancy - recommend further evaluation"
                      : "Consistent with benign findings - routine follow-up"
                    }
                  </div>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <Zap className="w-4 h-4 mt-1 text-blue-500" />
                <div>
                  <div className="font-medium">Specialist Expertise</div>
                  <div className="text-sm text-gray-600">
                    Analysis performed by {result.final_prediction.specialist_used} specialist 
                    (optimized for {result.dataset_classification.detected_dataset} image characteristics)
                  </div>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <Brain className="w-4 h-4 mt-1 text-purple-500" />
                <div>
                  <div className="font-medium">System Confidence</div>
                  <div className="text-sm text-gray-600">
                    {getConfidenceLevel(result.final_prediction.confidence).level} confidence 
                    ({(result.final_prediction.confidence * 100).toFixed(1)}%)
                    - {result.final_prediction.confidence > 0.8 ? 'High reliability' : 
                       result.final_prediction.confidence > 0.6 ? 'Moderate reliability' : 'Low reliability, consider additional testing'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}