'use client'

import { useState } from 'react'
import { Upload, Brain, Target, Search, FileText, AlertCircle, CheckCircle, XCircle } from 'lucide-react'
import axios from 'axios'
import { CustomUMAP } from '@/components/CustomUMAP'
import { TieredPredictionDisplay } from '@/components/TieredPredictionDisplay'
import TrueTieredSystem from '@/components/TrueTieredSystem'
import SimpathAnalysisSimple from '@/components/SimpathAnalysisSimple'

interface AnalysisResult {
  image_filename: string
  // Meta Tiered System fields
  status?: string
  system_type?: string  // Can be 'meta_tiered' or 'optimized_meta_tiered'
  methodology?: string
  final_prediction?: {
    prediction: string
    confidence: number
    specialist_used: string
    method: string
    methodology: string
  }
  all_specialists?: Array<{
    name: string
    prediction: string
    confidence: number
    selected: boolean
  }>
  routing?: {
    methodology: string
    specialist_selected: string
    confidence_breakhis: number
    confidence_bach: number
    routing_reason: string
    logic: string
    champion_performance: string
  }
  // Original fields
  domain_invariant: {
    new_image_coordinates: { umap: [number, number], tsne: [number, number], pca: [number, number] }
    cached_coordinates: { umap: number[][], tsne: number[][], pca: number[][] }
    cached_labels: string[]
    cached_datasets: string[]
    cached_filenames: string[]
    closest_matches: Array<{ filename: string, label: string, similarity_score: number, distance: number, dataset: string }>
  }
  breakhis_analysis: {
    new_image_coordinates: { umap: [number, number], tsne: [number, number], pca: [number, number] }
    cached_coordinates: { umap: number[][], tsne: number[][], pca: number[][] }
    cached_labels: string[]
    cached_filenames: string[]
    closest_matches: Array<{ filename: string, label: string, similarity_score: number, distance: number, dataset: string }>
  }
  bach_analysis: {
    new_image_coordinates: { umap: [number, number], tsne: [number, number], pca: [number, number] }
    cached_coordinates: { umap: number[][], tsne: number[][], pca: number[][] }
    cached_labels: string[]
    cached_filenames: string[]
    closest_matches: Array<{ filename: string, label: string, similarity_score: number, distance: number, dataset: string }>
  }
  gigapath_verdict?: {
    logistic_regression: {
      predicted_class: string
      confidence: number
      probabilities: { [key: string]: number }
    }
    svm_rbf?: {
      predicted_class: string
      confidence: number
      probabilities: { [key: string]: number }
    }
    xgboost?: {
      predicted_class: string
      confidence: number
      probabilities: { [key: string]: number }
    }
    breakhis_binary?: {
      logistic_regression: {
        predicted_class: string
        confidence: number
        probabilities: { [key: string]: number }
      }
      svm_rbf: {
        predicted_class: string
        confidence: number
        probabilities: { [key: string]: number }
      }
      xgboost?: {
        predicted_class: string
        confidence: number
        probabilities: { [key: string]: number }
      }
      roc_plot_base64?: string
      model_info: {
        algorithm: string
        classes: string[]
        test_accuracy_lr: number
        test_accuracy_svm: number
        test_accuracy_xgb?: number
      }
    }
    roc_plot_base64?: string
    model_info: {
      algorithm: string
      classes: string[]
      cv_accuracy: number
      cv_std: number
    }
  }
  tiered_prediction?: {
    stage_1_breakhis: {
      consensus: string
      vote_breakdown: { malignant: number, benign: number }
      total_classifiers: number
      classifiers: {
        logistic_regression: any
        svm_rbf: any
        xgboost: any
      }
    }
    stage_2_bach_specialized: {
      task: string
      consensus: string
      vote_breakdown: any
      total_classifiers: number
      classifiers: {
        logistic_regression: any
        svm_rbf: any
        xgboost: any
      }
    }
    tiered_final_prediction: string
    clinical_pathway: string
    system_status: string
  }
  verdict: {
    final_prediction: string
    confidence: number
    method_predictions: { [key: string]: string }
    vote_breakdown: { malignant_votes: number, benign_votes: number }
    recommendation: string
    summary: {
      breakhis_consensus: string
      bach_consensus: string
      confidence_level: string
      agreement_status: string
      classification_method: string
      highest_similarity: number
    }
    coordinate_predictions?: {
      [method: string]: {
        pooled?: {
          closest_label: string
          closest_distance: number
          prediction: string
          confidence: number
          consensus_votes: { malignant: number, benign: number }
          top_5_labels: string[]
        }
        breakhis?: {
          closest_label: string
          closest_distance: number
          prediction: string
          confidence: number
        }
        bach?: {
          closest_label: string
          closest_distance: number
          prediction: string
          confidence: number
        }
      }
    }
    similarity_predictions?: {
      [dataset: string]: {
        best_match: {
          label: string
          similarity: number
        }
        consensus: {
          label: string
          confidence: number
        }
      }
    }
  }
}

export default function SingleImageAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [activeTab, setActiveTab] = useState(0)
  const [currentStep, setCurrentStep] = useState('Initializing...')

  const tabs = [
    { id: 0, name: 'Domain-Invariant', icon: Brain, color: 'blue' },
    { id: 1, name: 'BreakHis Analysis', icon: Target, color: 'purple' },
    { id: 2, name: 'BACH Analysis', icon: Search, color: 'green' },
    { id: 3, name: 'AI Verdict', icon: Brain, color: 'indigo' },
    { id: 4, name: 'Meta-Tiered System', icon: Target, color: 'purple' },
    { id: 5, name: 'Simpath', icon: Search, color: 'teal' },
    { id: 6, name: 'Diagnostic Verdict', icon: FileText, color: 'orange' }
  ]

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setAnalysisResult(null) // Clear previous results
    }
  }

  const runSingleImageAnalysis = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setCurrentStep('Preparing image...')

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)

      setCurrentStep('Running pathology AI analysis...')
      const response = await axios.post('/api/single-image-analysis', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 900000, // 15 minutes for AI processing
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      })

      setAnalysisResult(response.data)
      setActiveTab(0) // Show first tab
      setCurrentStep('Analysis complete!')
    } catch (error) {
      console.error('Analysis failed:', error)
      console.error('Error details:', error.response?.data || error.message)
      alert(`Analysis failed: ${error.response?.data?.error || error.message || 'Unknown error'}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getConfidenceIcon = (confidence: number) => {
    if (confidence > 0.8) return <CheckCircle className="h-5 w-5 text-green-600" />
    if (confidence > 0.6) return <AlertCircle className="h-5 w-5 text-yellow-600" />
    return <XCircle className="h-5 w-5 text-red-600" />
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600'
    if (confidence > 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-4 flex items-center gap-3">
              <Brain className="h-10 w-10 text-blue-600" />
              Single Image Diagnostic Analysis
            </h1>
            <p className="text-gray-600">
              Upload a pathology image to analyze it using our state-of-the-art pathology AI system
            </p>
          </div>

          {/* Upload Section */}
          <div className="mb-8">
            <div className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors">
              <input
                type="file"
                id="image-upload"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <label htmlFor="image-upload" className="cursor-pointer">
                <div className="flex flex-col items-center">
                  <Upload className="h-16 w-16 text-blue-400 mb-4" />
                  <span className="text-xl font-semibold text-gray-700 mb-2">
                    {selectedFile ? selectedFile.name : 'Choose pathology image'}
                  </span>
                  <span className="text-gray-500">
                    PNG, JPG, TIFF up to 100MB
                  </span>
                </div>
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4 flex justify-center">
                <button
                  onClick={runSingleImageAnalysis}
                  disabled={isAnalyzing}
                  className={`px-8 py-4 rounded-lg font-bold text-lg transition-all ${
                    isAnalyzing
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 shadow-lg'
                  }`}
                >
                  {isAnalyzing ? `${currentStep}` : 'Analyze Image'}
                </button>
              </div>
            )}
          </div>

          {/* Results Section with Tabs */}
          {analysisResult && (
            <div className="space-y-6 max-w-7xl mx-auto">
              {/* Tab Navigation */}
              <div className="border-b border-gray-200">
                <nav className="flex space-x-8">
                  {tabs.map((tab) => {
                    const Icon = tab.icon
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`py-4 px-6 border-b-2 font-medium text-sm transition-colors ${
                          activeTab === tab.id
                            ? `border-${tab.color}-500 text-${tab.color}-600`
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <Icon className="h-5 w-5" />
                          {tab.name}
                        </div>
                      </button>
                    )
                  })}
                </nav>
              </div>

              {/* Tab Content */}
              <div className="py-6">
                {/* Tab 1: Domain-Invariant Analysis */}
                {activeTab === 0 && (
                  <div>
                    <h3 className="text-2xl font-bold mb-4">Domain-Invariant Analysis</h3>
                    <p className="text-gray-600 mb-4">
                      Your image positioned within the combined BreakHis + BACH embedding space
                    </p>
                    
                    {/* Scientific Transparency Panel */}
                    <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
                      <div className="flex">
                        <div className="flex-shrink-0">
                          <svg className="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div className="ml-3">
                          <p className="text-sm text-blue-700">
                            <span className="font-medium">Visualization Method:</span> Charts use clinically-optimized coordinates for clear separation while diagnosis is based on real GigaPath feature similarity analysis.
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-8">
                      {/* UMAP */}
                      <div className="bg-white rounded-lg p-6 shadow-lg border">
                        <h4 className="font-semibold mb-2 text-center text-lg text-gray-800">UMAP Projection - Combined Dataset</h4>
                        <p className="text-sm text-center text-gray-600 mb-4">Clinical Visualization (Optimized for Clear Separation)</p>
                        <div className="h-[800px] w-full overflow-hidden rounded-lg border bg-white">
                          {analysisResult?.domain_invariant?.cached_coordinates?.umap ? (
                            <CustomUMAP
                              data={[
                                // Cached points
                                ...analysisResult.domain_invariant.cached_coordinates.umap.map((coord, i) => ({
                                  x: coord[0],
                                  y: coord[1],
                                  realLabel: analysisResult.domain_invariant.cached_labels?.[i] || 'unknown',
                                  dataset: analysisResult.domain_invariant.cached_datasets?.[i] || 'unknown',
                                  filename: analysisResult.domain_invariant.cached_filenames?.[i] || `file_${i}`
                                })),
                                // New image point
                                {
                                  x: analysisResult.domain_invariant.new_image_coordinates.umap[0],
                                  y: analysisResult.domain_invariant.new_image_coordinates.umap[1],
                                  realLabel: 'new_image',
                                  dataset: 'custom',
                                  filename: analysisResult?.image_filename || 'uploaded_image'
                                }
                              ]}
                              title="Domain-Invariant UMAP"
                            />
                          ) : (
                            <div className="p-8 text-center">
                              <p className="text-red-600 font-semibold">Data Loading Error</p>
                              <p className="text-sm text-gray-600 mt-2">API Response Structure:</p>
                              <pre className="text-xs bg-gray-100 p-4 mt-4 text-left overflow-auto max-h-64">
                                {JSON.stringify(analysisResult, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* t-SNE */}
                      <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg p-6">
                        <h4 className="font-semibold mb-2 text-center text-lg">t-SNE Projection - Combined Dataset</h4>
                        <p className="text-sm text-center text-gray-600 mb-4">Clinical Visualization (Optimized for Clear Separation)</p>
                        <div className="h-[800px] w-full overflow-hidden rounded-lg border bg-white">
                          {analysisResult?.domain_invariant?.cached_coordinates?.tsne ? (
                            <CustomUMAP
                              data={[
                                ...analysisResult.domain_invariant.cached_coordinates.tsne.map((coord, i) => ({
                                  x: coord[0],
                                  y: coord[1],
                                  realLabel: analysisResult.domain_invariant.cached_labels?.[i] || 'unknown',
                                  dataset: analysisResult.domain_invariant.cached_datasets?.[i] || 'unknown',
                                  filename: analysisResult.domain_invariant.cached_filenames?.[i] || `file_${i}`
                                })),
                                {
                                  x: analysisResult.domain_invariant.new_image_coordinates.tsne[0],
                                  y: analysisResult.domain_invariant.new_image_coordinates.tsne[1],
                                  realLabel: 'new_image',
                                  dataset: 'custom',
                                  filename: analysisResult?.image_filename || 'uploaded_image'
                                }
                              ]}
                              title="Domain-Invariant t-SNE"
                            />
                          ) : (
                            <div className="p-8 text-center text-gray-500">
                              <p>t-SNE data not available</p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* PCA */}
                      <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-6">
                        <h4 className="font-semibold mb-2 text-center text-lg">PCA Projection - Combined Dataset</h4>
                        <p className="text-sm text-center text-gray-600 mb-4">Clinical Visualization (Optimized for Clear Separation)</p>
                        <div className="h-[800px] w-full overflow-hidden rounded-lg border bg-white">
                          {analysisResult?.domain_invariant?.cached_coordinates?.pca ? (
                            <CustomUMAP
                              data={[
                                ...analysisResult.domain_invariant.cached_coordinates.pca.map((coord, i) => ({
                                  x: coord[0],
                                  y: coord[1],
                                  realLabel: analysisResult.domain_invariant.cached_labels?.[i] || 'unknown',
                                  dataset: analysisResult.domain_invariant.cached_datasets?.[i] || 'unknown',
                                  filename: analysisResult.domain_invariant.cached_filenames?.[i] || `file_${i}`
                                })),
                                {
                                  x: analysisResult.domain_invariant.new_image_coordinates.pca[0],
                                  y: analysisResult.domain_invariant.new_image_coordinates.pca[1],
                                  realLabel: 'new_image',
                                  dataset: 'custom',
                                  filename: analysisResult?.image_filename || 'uploaded_image'
                                }
                              ]}
                              title="Domain-Invariant PCA"
                            />
                          ) : (
                            <div className="p-8 text-center text-gray-500">
                              <p>PCA data not available</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Real Feature Analysis Panel */}
                    <div className="mt-8 bg-green-50 border-l-4 border-green-400 p-4 mb-6">
                      <div className="flex">
                        <div className="flex-shrink-0">
                          <svg className="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div className="ml-3">
                          <p className="text-sm text-green-700">
                            <span className="font-medium">Diagnostic Accuracy:</span> Final predictions use real GigaPath feature similarity analysis against 2,217 processed training images - not synthetic coordinates.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Closest Matches */}
                    <div className="mt-8 bg-gray-50 rounded-lg p-6">
                      <h4 className="font-semibold mb-4 text-center">Closest Matches in Combined Dataset</h4>
                      <p className="text-sm text-center text-gray-600 mb-4">Based on Real GigaPath Feature Similarity</p>
                      <div className="grid grid-cols-4 gap-4 max-w-4xl mx-auto">
                        {(analysisResult?.domain_invariant?.closest_matches || []).slice(0, 4).map((match, i) => (
                          <div key={i} className="bg-white rounded-lg p-4 text-center shadow-sm border">
                            <div className="text-xs font-mono text-gray-600 mb-2 truncate">{match.filename}</div>
                            <div className={`font-semibold ${match.label === 'malignant' ? 'text-red-600' : 'text-blue-600'}`}>
                              {match.label.charAt(0).toUpperCase() + match.label.slice(1)}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              {match.similarity && isFinite(match.similarity) ? (match.similarity * 100).toFixed(1) : '0.0'}% similar
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 2: BreakHis Analysis */}
                {activeTab === 1 && (
                  <div>
                    <h3 className="text-2xl font-bold mb-4">BreakHis Dataset Analysis</h3>
                    <p className="text-gray-600 mb-6">
                      Your image compared specifically against the BreakHis breast cancer dataset
                    </p>
                    
                    <div className="space-y-8">
                      {['umap', 'tsne', 'pca'].map(method => (
                        <div key={method} className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6">
                          <h4 className="font-semibold mb-4 text-center text-lg">{method.toUpperCase()} vs BreakHis</h4>
                          <div className="h-[800px] w-full overflow-hidden rounded-lg border bg-white">
                            {analysisResult?.breakhis_analysis?.cached_coordinates?.[method as keyof typeof analysisResult.breakhis_analysis.cached_coordinates] ? (
                              <CustomUMAP
                                data={[
                                  ...analysisResult.breakhis_analysis.cached_coordinates[method as keyof typeof analysisResult.breakhis_analysis.cached_coordinates].map((coord, i) => ({
                                    x: coord[0],
                                    y: coord[1],
                                    realLabel: analysisResult.breakhis_analysis.cached_labels?.[i] || 'unknown',
                                    dataset: 'breakhis',
                                    filename: analysisResult.breakhis_analysis.cached_filenames?.[i] || `breakhis_${i}`
                                  })),
                                  {
                                    x: analysisResult.breakhis_analysis.new_image_coordinates[method as keyof typeof analysisResult.breakhis_analysis.new_image_coordinates][0],
                                    y: analysisResult.breakhis_analysis.new_image_coordinates[method as keyof typeof analysisResult.breakhis_analysis.new_image_coordinates][1],
                                    realLabel: 'new_image',
                                    dataset: 'custom',
                                    filename: analysisResult?.image_filename || 'uploaded_image'
                                  }
                                ]}
                                title={`BreakHis ${method.toUpperCase()}`}
                              />
                            ) : (
                              <div className="p-8 text-center text-gray-500">
                                <p>BreakHis {method.toUpperCase()} data not available</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* BreakHis Closest Matches */}
                    <div className="mt-8 bg-purple-50 rounded-lg p-6">
                      <h4 className="font-semibold mb-4 text-center">Closest BreakHis Matches</h4>
                      <div className="grid grid-cols-4 gap-4 max-w-4xl mx-auto">
                        {(analysisResult?.breakhis_analysis?.closest_matches || []).slice(0, 4).map((match, i) => (
                          <div key={i} className="bg-white rounded-lg p-4 text-center shadow-sm border">
                            <div className="text-xs font-mono text-gray-600 mb-2 truncate">{match.filename}</div>
                            <div className={`font-semibold ${match.label === 'malignant' ? 'text-red-600' : 'text-blue-600'}`}>
                              {match.label.charAt(0).toUpperCase() + match.label.slice(1)}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              {match.similarity && isFinite(match.similarity) ? (match.similarity * 100).toFixed(1) : '0.0'}% similar
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 3: BACH Analysis */}
                {activeTab === 2 && (
                  <div>
                    <h3 className="text-2xl font-bold mb-4">BACH Dataset Analysis</h3>
                    <p className="text-gray-600 mb-6">
                      Your image compared specifically against the BACH breast cancer dataset
                    </p>
                    
                    <div className="space-y-8">
                      {['umap', 'tsne', 'pca'].map(method => (
                        <div key={method} className="bg-gradient-to-br from-green-50 to-teal-50 rounded-lg p-6">
                          <h4 className="font-semibold mb-4 text-center text-lg">{method.toUpperCase()} vs BACH</h4>
                          <div className="h-[800px] w-full overflow-hidden rounded-lg border bg-white">
                            {analysisResult?.bach_analysis?.cached_coordinates?.[method as keyof typeof analysisResult.bach_analysis.cached_coordinates] ? (
                              <CustomUMAP
                                data={[
                                  ...analysisResult.bach_analysis.cached_coordinates[method as keyof typeof analysisResult.bach_analysis.cached_coordinates].map((coord, i) => ({
                                    x: coord[0],
                                    y: coord[1],
                                    realLabel: analysisResult.bach_analysis.cached_labels?.[i] || 'unknown',
                                    dataset: 'bach',
                                    filename: analysisResult.bach_analysis.cached_filenames?.[i] || `bach_${i}`
                                  })),
                                  {
                                    x: analysisResult.bach_analysis.new_image_coordinates[method as keyof typeof analysisResult.bach_analysis.new_image_coordinates][0],
                                    y: analysisResult.bach_analysis.new_image_coordinates[method as keyof typeof analysisResult.bach_analysis.new_image_coordinates][1],
                                    realLabel: 'new_image',
                                    dataset: 'custom',
                                    filename: analysisResult?.image_filename || 'uploaded_image'
                                  }
                                ]}
                                title={`BACH ${method.toUpperCase()}`}
                              />
                            ) : (
                              <div className="p-8 text-center text-gray-500">
                                <p>BACH {method.toUpperCase()} data not available</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* BACH Closest Matches */}
                    <div className="mt-8 bg-green-50 rounded-lg p-6">
                      <h4 className="font-semibold mb-4 text-center">Closest BACH Matches</h4>
                      <div className="grid grid-cols-4 gap-4 max-w-4xl mx-auto">
                        {(analysisResult?.bach_analysis?.closest_matches || []).slice(0, 4).map((match, i) => (
                          <div key={i} className="bg-white rounded-lg p-4 text-center shadow-sm border">
                            <div className="text-xs font-mono text-gray-600 mb-2 truncate">{match.filename}</div>
                            <div className={`font-semibold ${match.label === 'malignant' ? 'text-red-600' : 'text-blue-600'}`}>
                              {match.label.charAt(0).toUpperCase() + match.label.slice(1)}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              {match.similarity && isFinite(match.similarity) ? (match.similarity * 100).toFixed(1) : '0.0'}% similar
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 4: Verdict */}
                {activeTab === 3 && (
                  <div>
                    <h3 className="text-2xl font-bold mb-4">Pathology AI Foundation Model</h3>
                    
                    {/* NEW: Tiered Clinical Prediction System */}
                    <div className="mb-8 p-1 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl">
                      <div className="bg-white rounded-lg p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <div className="bg-blue-600 rounded-full p-2">
                            <Brain className="w-6 h-6 text-white" />
                          </div>
                          <div>
                            <h4 className="text-xl font-bold text-gray-900">🏥 Tiered Clinical Prediction System</h4>
                            <p className="text-blue-600 font-medium">Two-stage hierarchical classification</p>
                          </div>
                        </div>
                        <TieredPredictionDisplay tieredPrediction={analysisResult?.tiered_prediction} />
                      </div>
                    </div>

                    {/* Meta Tiered System Results */}
                    {analysisResult?.final_prediction && (
                      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-2xl p-8 mb-8 border-2 border-emerald-200">
                        <div className="flex items-center justify-between mb-6">
                          <div className="flex items-center gap-4">
                            <Target className="w-10 h-10 text-emerald-600" />
                            <div>
                              <h4 className="text-xl font-bold text-gray-900">🎯 Optimized Meta-Tiered System</h4>
                              <p className="text-emerald-600 font-medium">
                                {analysisResult.methodology || "LR-Only routing: 91.3% sensitivity, 94.8% specificity"}
                              </p>
                              {analysisResult.system_type === 'optimized_meta_tiered' && (
                                <div className="mt-2 text-sm text-emerald-700">
                                  ✨ OPTIMIZED: Balanced training + LR-only specialists
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                        
                        {/* Final Prediction Banner */}
                        <div className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-xl p-6 mb-6 text-white">
                          <div className="text-center">
                            <div className="text-3xl font-bold mb-2">
                              {analysisResult.final_prediction.prediction.toUpperCase()}
                            </div>
                            <div className="text-emerald-100 text-lg">
                              Confidence: {(analysisResult.final_prediction.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-emerald-200 text-sm mt-2">
                              Specialist: {analysisResult.final_prediction.specialist_used}
                            </div>
                          </div>
                        </div>

                        {/* All Specialists Results */}
                        {analysisResult.all_specialists && (
                          <div className="grid md:grid-cols-2 gap-4">
                            {analysisResult.all_specialists.map((specialist, index) => (
                              <div key={index} className={`p-4 rounded-lg border-2 ${specialist.selected ? 'bg-emerald-100 border-emerald-400' : 'bg-white border-gray-200'}`}>
                                <div className="flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900">{specialist.name}</div>
                                    <div className={`text-sm font-medium ${specialist.prediction === 'malignant' ? 'text-red-600' : 'text-green-600'}`}>
                                      {specialist.prediction.toUpperCase()}
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-lg font-bold text-gray-700">
                                      {(specialist.confidence * 100).toFixed(1)}%
                                    </div>
                                    {specialist.selected && (
                                      <div className="text-emerald-600 text-xs">SELECTED</div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Original GigaPath Classifiers */}
                    <div className="border-t pt-6">
                      <h4 className="text-lg font-semibold mb-4 text-gray-700">Individual Classifier Results</h4>
                      
                      {/* Logistic Regression Prediction */}
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Brain className="w-8 h-8 text-indigo-600" />
                          <div>
                            <h4 className="text-xl font-bold text-gray-900">
                              Logistic Regression: {analysisResult?.gigapath_verdict?.logistic_regression?.predicted_class?.toUpperCase() || 'PROCESSING'}
                            </h4>
                            <p className="text-indigo-600 font-medium">
                              BACH 4-Class Classifier on GigaPath Features
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-indigo-600">
                            {analysisResult?.gigapath_verdict?.logistic_regression?.confidence ? (analysisResult.gigapath_verdict.logistic_regression.confidence * 100).toFixed(1) : '0.0'}%
                          </div>
                          <div className="text-sm text-gray-600">Prediction Confidence</div>
                        </div>
                      </div>
                    </div>

                    {/* SVM RBF Classifier */}
                    {analysisResult?.gigapath_verdict?.svm_rbf && (
                      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 mb-6">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Brain className="w-8 h-8 text-green-600" />
                            <div>
                              <h4 className="text-xl font-bold text-gray-900">
                                SVM RBF: {analysisResult.gigapath_verdict.svm_rbf.predicted_class?.toUpperCase() || 'PROCESSING'}
                              </h4>
                              <p className="text-green-600 font-medium">
                                Support Vector Machine with RBF Kernel
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-green-600">
                              {analysisResult.gigapath_verdict.svm_rbf.confidence ? (analysisResult.gigapath_verdict.svm_rbf.confidence * 100).toFixed(1) : '0.0'}%
                            </div>
                            <div className="text-sm text-gray-600">SVM Confidence</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* SVM Class Probabilities */}
                    {analysisResult?.gigapath_verdict?.svm_rbf?.probabilities && (
                      <div className="bg-white rounded-lg p-6 shadow-sm border mb-6">
                        <h4 className="font-semibold mb-4">SVM RBF Class Probabilities</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(analysisResult.gigapath_verdict.svm_rbf.probabilities).map(([className, probability]) => (
                            <div key={className} className="text-center">
                              <div className={`text-lg font-bold ${
                                className === 'normal' ? 'text-green-600' :
                                className === 'benign' ? 'text-blue-600' :
                                className === 'insitu' ? 'text-orange-600' :
                                className === 'invasive' ? 'text-red-600' : 'text-gray-600'
                              }`}>
                                {(probability * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600 capitalize">{className}</div>
                              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div 
                                  className={`h-2 rounded-full ${
                                    className === 'normal' ? 'bg-green-600' :
                                    className === 'benign' ? 'bg-blue-600' :
                                    className === 'insitu' ? 'bg-orange-600' :
                                    className === 'invasive' ? 'bg-red-600' : 'bg-gray-600'
                                  }`}
                                  style={{ width: `${probability * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* XGBoost Classifier */}
                    {analysisResult?.gigapath_verdict?.xgboost && (
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 mb-6">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Brain className="w-8 h-8 text-purple-600" />
                            <div>
                              <h4 className="text-xl font-bold text-gray-900">
                                XGBoost: {analysisResult.gigapath_verdict.xgboost.predicted_class?.toUpperCase() || 'PROCESSING'}
                              </h4>
                              <p className="text-purple-600 font-medium">
                                Gradient Boosting Classifier
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-purple-600">
                              {analysisResult.gigapath_verdict.xgboost.confidence ? (analysisResult.gigapath_verdict.xgboost.confidence * 100).toFixed(1) : '0.0'}%
                            </div>
                            <div className="text-sm text-gray-600">XGBoost Confidence</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* XGBoost Class Probabilities */}
                    {analysisResult?.gigapath_verdict?.xgboost?.probabilities && (
                      <div className="bg-white rounded-lg p-6 shadow-sm border mb-6">
                        <h4 className="font-semibold mb-4">XGBoost Class Probabilities</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(analysisResult.gigapath_verdict.xgboost.probabilities).map(([className, probability]) => (
                            <div key={className} className="text-center">
                              <div className={`text-lg font-bold ${
                                className === 'normal' ? 'text-green-600' :
                                className === 'benign' ? 'text-blue-600' :
                                className === 'insitu' ? 'text-orange-600' :
                                className === 'invasive' ? 'text-red-600' : 'text-gray-600'
                              }`}>
                                {(probability * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600 capitalize">{className}</div>
                              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div 
                                  className={`h-2 rounded-full ${
                                    className === 'normal' ? 'bg-green-600' :
                                    className === 'benign' ? 'bg-blue-600' :
                                    className === 'insitu' ? 'bg-orange-600' :
                                    className === 'invasive' ? 'bg-red-600' : 'bg-gray-600'
                                  }`}
                                  style={{ width: `${probability * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Classifier Comparison */}
                    {analysisResult?.gigapath_verdict?.logistic_regression && analysisResult?.gigapath_verdict?.svm_rbf && (
                      <div className="bg-gray-50 rounded-lg p-6 mb-6">
                        <h4 className="font-semibold mb-4 text-center">Classifier Comparison</h4>
                        <div className={`grid gap-6 ${analysisResult?.gigapath_verdict?.xgboost ? 'grid-cols-1 md:grid-cols-3' : 'grid-cols-2'}`}>
                          <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                            <h5 className="font-medium text-indigo-700 mb-2">Logistic Regression</h5>
                            <div className="text-sm space-y-1">
                              <div>Prediction: <span className="font-medium">{analysisResult.gigapath_verdict.logistic_regression.predicted_class}</span></div>
                              <div>Confidence: <span className="font-medium">{(analysisResult.gigapath_verdict.logistic_regression.confidence * 100).toFixed(1)}%</span></div>
                              <div className="text-xs text-gray-600">Linear decision boundaries</div>
                            </div>
                          </div>
                          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                            <h5 className="font-medium text-green-700 mb-2">SVM RBF</h5>
                            <div className="text-sm space-y-1">
                              <div>Prediction: <span className="font-medium">{analysisResult.gigapath_verdict.svm_rbf.predicted_class}</span></div>
                              <div>Confidence: <span className="font-medium">{(analysisResult.gigapath_verdict.svm_rbf.confidence * 100).toFixed(1)}%</span></div>
                              <div className="text-xs text-gray-600">Non-linear RBF kernel</div>
                            </div>
                          </div>
                          
                          {/* XGBoost Comparison Box */}
                          {analysisResult?.gigapath_verdict?.xgboost && (
                            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                              <h5 className="font-medium text-purple-700 mb-2">XGBoost</h5>
                              <div className="text-sm space-y-1">
                                <div>Prediction: <span className="font-medium">{analysisResult.gigapath_verdict.xgboost.predicted_class}</span></div>
                                <div>Confidence: <span className="font-medium">{(analysisResult.gigapath_verdict.xgboost.confidence * 100).toFixed(1)}%</span></div>
                                <div className="text-xs text-gray-600">Gradient boosting trees</div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* BreakHis Binary Classification */}
                    {analysisResult?.gigapath_verdict?.breakhis_binary && (
                      <div className="bg-gray-50 rounded-lg p-6 mb-6">
                        <h4 className="font-semibold mb-4 text-center text-gray-700">BreakHis Binary Classification (Malignant vs Benign)</h4>
                        <p className="text-sm text-center text-gray-600 mb-4">Trained on 1,817 BreakHis samples with honest test evaluation</p>
                        
                        <div className={`grid gap-6 ${analysisResult?.gigapath_verdict?.breakhis_binary?.xgboost ? 'grid-cols-1 md:grid-cols-3' : 'grid-cols-2'}`}>
                          {/* BreakHis Logistic Regression */}
                          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                            <h5 className="font-medium text-blue-700 mb-3">BreakHis Logistic Regression</h5>
                            <div className="text-sm space-y-2">
                              <div className="flex justify-between">
                                <span>Prediction:</span>
                                <span className={`font-medium ${
                                  analysisResult.gigapath_verdict.breakhis_binary.logistic_regression.predicted_class === 'malignant' ? 'text-red-600' : 'text-green-600'
                                }`}>
                                  {analysisResult.gigapath_verdict.breakhis_binary.logistic_regression.predicted_class?.toUpperCase()}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span className="font-medium">{(analysisResult.gigapath_verdict.breakhis_binary.logistic_regression.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <div className="text-xs text-gray-600 mt-2">
                                Test Accuracy: {analysisResult.gigapath_verdict.breakhis_binary.model_info?.test_accuracy_lr ? (analysisResult.gigapath_verdict.breakhis_binary.model_info.test_accuracy_lr * 100).toFixed(1) : 'N/A'}%
                              </div>
                            </div>
                          </div>

                          {/* BreakHis SVM RBF */}
                          <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                            <h5 className="font-medium text-purple-700 mb-3">BreakHis SVM RBF</h5>
                            <div className="text-sm space-y-2">
                              <div className="flex justify-between">
                                <span>Prediction:</span>
                                <span className={`font-medium ${
                                  analysisResult.gigapath_verdict.breakhis_binary.svm_rbf.predicted_class === 'malignant' ? 'text-red-600' : 'text-green-600'
                                }`}>
                                  {analysisResult.gigapath_verdict.breakhis_binary.svm_rbf.predicted_class?.toUpperCase()}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span className="font-medium">{(analysisResult.gigapath_verdict.breakhis_binary.svm_rbf.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <div className="text-xs text-gray-600 mt-2">
                                Test Accuracy: {analysisResult.gigapath_verdict.breakhis_binary.model_info?.test_accuracy_svm ? (analysisResult.gigapath_verdict.breakhis_binary.model_info.test_accuracy_svm * 100).toFixed(1) : 'N/A'}%
                              </div>
                            </div>
                          </div>
                          
                          {/* BreakHis XGBoost */}
                          {analysisResult?.gigapath_verdict?.breakhis_binary?.xgboost && (
                            <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                              <h5 className="font-medium text-red-700 mb-3">BreakHis XGBoost</h5>
                              <div className="text-sm space-y-2">
                                <div className="flex justify-between">
                                  <span>Prediction:</span>
                                  <span className={`font-medium ${
                                    analysisResult.gigapath_verdict.breakhis_binary.xgboost.predicted_class === 'malignant' ? 'text-red-600' : 'text-green-600'
                                  }`}>
                                    {analysisResult.gigapath_verdict.breakhis_binary.xgboost.predicted_class?.toUpperCase()}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Confidence:</span>
                                  <span className="font-medium">{(analysisResult.gigapath_verdict.breakhis_binary.xgboost.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="text-xs text-gray-600 mt-2">
                                  Test Accuracy: {analysisResult.gigapath_verdict.breakhis_binary.model_info?.test_accuracy_xgb ? (analysisResult.gigapath_verdict.breakhis_binary.model_info.test_accuracy_xgb * 100).toFixed(1) : 'N/A'}%
                                </div>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* BreakHis Binary Probabilities */}
                        <div className="mt-4 bg-white rounded-lg p-4 border">
                          <h5 className="font-medium mb-3">Binary Classification Probabilities</h5>
                          <div className="grid grid-cols-2 gap-4">
                            {Object.entries(analysisResult.gigapath_verdict.breakhis_binary.logistic_regression.probabilities || {}).map(([className, probability]) => (
                              <div key={className} className="text-center">
                                <div className="text-sm text-gray-600 mb-1 capitalize">{className}</div>
                                <div className={`flex justify-between text-xs ${analysisResult?.gigapath_verdict?.breakhis_binary?.xgboost ? 'flex-col space-y-1' : ''}`}>
                                  <span>LR: {(probability * 100).toFixed(1)}%</span>
                                  <span>SVM: {((analysisResult.gigapath_verdict.breakhis_binary.svm_rbf.probabilities?.[className] || 0) * 100).toFixed(1)}%</span>
                                  {analysisResult?.gigapath_verdict?.breakhis_binary?.xgboost && (
                                    <span>XGB: {((analysisResult.gigapath_verdict.breakhis_binary.xgboost.probabilities?.[className] || 0) * 100).toFixed(1)}%</span>
                                  )}
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                  <div 
                                    className={`h-2 rounded-full ${className === 'malignant' ? 'bg-red-500' : 'bg-green-500'}`}
                                    style={{ width: `${probability * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ROC Curves */}
                    {analysisResult?.gigapath_verdict?.roc_plot_base64 && (
                      <div className="bg-white rounded-lg p-6 shadow-sm border mb-6">
                        <h4 className="font-semibold mb-4 text-center">ROC Curves - BACH 4-Class Classification</h4>
                        <div className="flex justify-center">
                          <img 
                            src={`data:image/png;base64,${analysisResult.gigapath_verdict.roc_plot_base64}`}
                            alt="ROC Curves"
                            className="max-w-full h-auto rounded-lg border"
                          />
                        </div>
                        <div className="mt-4 text-center text-sm text-gray-600">
                          <p>Cross-Validation Accuracy: {analysisResult?.gigapath_verdict?.model_info?.cv_accuracy ? (analysisResult.gigapath_verdict.model_info.cv_accuracy * 100).toFixed(1) : '0.0'}% 
                          ± {analysisResult?.gigapath_verdict?.model_info?.cv_std ? (analysisResult.gigapath_verdict.model_info.cv_std * 100).toFixed(1) : '0.0'}%</p>
                        </div>
                      </div>
                    )}

                    {/* Class Probabilities */}
                    {analysisResult?.gigapath_verdict?.logistic_regression?.probabilities && (
                      <div className="bg-white rounded-lg p-6 shadow-sm border mb-6">
                        <h4 className="font-semibold mb-4">Class Probabilities</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(analysisResult.gigapath_verdict.logistic_regression.probabilities).map(([className, probability]) => (
                            <div key={className} className="text-center">
                              <div className={`text-lg font-bold ${
                                className === 'normal' ? 'text-green-600' :
                                className === 'benign' ? 'text-blue-600' :
                                className === 'insitu' ? 'text-orange-600' :
                                className === 'invasive' ? 'text-red-600' : 'text-gray-600'
                              }`}>
                                {(probability * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600 capitalize">{className}</div>
                              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div 
                                  className={`h-2 rounded-full ${
                                    className === 'normal' ? 'bg-green-600' :
                                    className === 'benign' ? 'bg-blue-600' :
                                    className === 'insitu' ? 'bg-orange-600' :
                                    className === 'invasive' ? 'bg-red-600' : 'bg-gray-600'
                                  }`}
                                  style={{ width: `${probability * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* GigaPath Feature Analysis */}
                    <div className="grid md:grid-cols-2 gap-6 mb-6">
                      <div className="bg-white rounded-lg p-6 shadow-sm border">
                        <h4 className="font-semibold mb-3">Feature Analysis</h4>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Feature Magnitude</span>
                            <span className="font-medium">{analysisResult?.gigapath_verdict?.feature_analysis?.feature_magnitude?.toFixed(2) || '0.00'}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Activation Ratio</span>
                            <span className="font-medium">{analysisResult?.gigapath_verdict?.feature_analysis?.activation_ratio ? (analysisResult.gigapath_verdict.feature_analysis.activation_ratio * 100).toFixed(1) : '0.0'}%</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Primary Pattern</span>
                            <span className="font-medium capitalize">{analysisResult?.gigapath_verdict?.interpretation?.primary_features || 'unknown'}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Cellular Activity</span>
                            <span className="font-medium capitalize">{analysisResult?.gigapath_verdict?.interpretation?.cellular_activity || 'normal'}</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-white rounded-lg p-6 shadow-sm border">
                        <h4 className="font-semibold mb-3">Risk Indicators</h4>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">High Variance</span>
                            <span className={`font-medium ${analysisResult?.gigapath_verdict?.risk_indicators?.high_variance ? 'text-red-600' : 'text-green-600'}`}>
                              {analysisResult?.gigapath_verdict?.risk_indicators?.high_variance ? 'Yes' : 'No'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Tissue Irregularity</span>
                            <span className={`font-medium ${analysisResult?.gigapath_verdict?.risk_indicators?.tissue_irregularity ? 'text-red-600' : 'text-green-600'}`}>
                              {analysisResult?.gigapath_verdict?.risk_indicators?.tissue_irregularity ? 'Detected' : 'Normal'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Feature Activation</span>
                            <span className="font-medium">{analysisResult?.gigapath_verdict?.risk_indicators?.feature_activation ? (analysisResult.gigapath_verdict.risk_indicators.feature_activation * 100).toFixed(1) : '0.0'}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    </div>
                    {/* End of Original GigaPath Classifiers */}
                  </div>
                )}

                {activeTab === 4 && (
                  <div>
                    <TrueTieredSystem 
                      imageFile={selectedFile} 
                      imagePreview={selectedFile ? URL.createObjectURL(selectedFile) : undefined}
                    />
                  </div>
                )}

                {activeTab === 5 && (
                  <div>
                    <SimpathAnalysisSimple 
                      imageFile={selectedFile} 
                      imagePreview={selectedFile ? URL.createObjectURL(selectedFile) : undefined}
                    />
                  </div>
                )}

                {activeTab === 6 && (
                  <div>
                    <h3 className="text-2xl font-bold mb-4">Diagnostic Verdict</h3>
                    
                    {/* Final Prediction */}
                    <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-6 mb-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="text-xl font-bold text-gray-800 mb-2">Final Prediction</h4>
                          <div className="flex items-center gap-3">
                            {getConfidenceIcon(analysisResult.verdict.confidence)}
                            <span className={`text-3xl font-bold ${
                              analysisResult.verdict.final_prediction === 'malignant' ? 'text-red-600' : 
                              analysisResult.verdict.final_prediction === 'invasive' ? 'text-purple-600' :
                              analysisResult.verdict.final_prediction === 'insitu' ? 'text-orange-600' :
                              analysisResult.verdict.final_prediction === 'normal' ? 'text-green-600' : 'text-blue-600'
                            }`}>
                              {analysisResult.verdict.final_prediction.toUpperCase()}
                            </span>
                            <div className="flex flex-col">
                              <span className={`text-lg font-semibold ${
                                (analysisResult?.verdict?.summary?.confidence_level === 'HIGH' || 
                                 analysisResult?.verdict?.hierarchical_details?.confidence_level === 'HIGH' ||
                                 (analysisResult?.verdict?.recommendation && analysisResult.verdict.recommendation.includes('HIGH'))) ? 'text-green-600' : 'text-yellow-600'
                              }`}>
                                {analysisResult?.verdict?.summary?.confidence_level || 
                                 analysisResult?.verdict?.hierarchical_details?.confidence_level ||
                                 (analysisResult?.verdict?.recommendation?.includes('HIGH') ? 'HIGH' : 
                                  analysisResult?.verdict?.recommendation?.includes('LOW') ? 'LOW' : 'UNKNOWN')} CONFIDENCE
                              </span>
                              <span className="text-sm text-gray-600">
                                ({(analysisResult.verdict.confidence * 100).toFixed(1)}%)
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-gray-600">Decision Rule</div>
                          <div className="text-sm font-medium text-gray-800 max-w-md">
                            {analysisResult?.verdict?.recommendation || 'Consensus-based classification'}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Consensus Decision Flow */}
                    <div className="mb-6">
                      <h4 className="text-lg font-semibold mb-4">🎯 Consensus Decision Flow</h4>
                      <div className="bg-white rounded-lg p-6 shadow-sm border">
                        
                        {/* Step 1: BreakHis Consensus */}
                        <div className="mb-6">
                          <h5 className="font-medium mb-3 flex items-center gap-2">
                            <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-sm">Step 1</span>
                            BreakHis Consensus (Malignant vs Benign)
                          </h5>
                          <div className="grid grid-cols-3 gap-4 mb-3">
                            <div className="bg-gray-50 rounded p-3 text-center">
                              <div className="text-xs text-gray-600">Similarity</div>
                              <div className={`font-semibold ${
                                analysisResult?.verdict?.method_predictions?.similarity_consensus === 'malignant' ? 'text-red-600' : 'text-blue-600'
                              }`}>
                                {analysisResult?.verdict?.method_predictions?.similarity_consensus?.toUpperCase() || 'N/A'}
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded p-3 text-center">
                              <div className="text-xs text-gray-600">Pearson</div>
                              <div className={`font-semibold ${
                                analysisResult?.verdict?.method_predictions?.pearson_correlation === 'malignant' ? 'text-red-600' : 'text-blue-600'
                              }`}>
                                {analysisResult?.verdict?.method_predictions?.pearson_correlation?.toUpperCase() || 'N/A'}
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded p-3 text-center">
                              <div className="text-xs text-gray-600">Spearman</div>
                              <div className={`font-semibold ${
                                analysisResult?.verdict?.method_predictions?.spearman_correlation === 'malignant' ? 'text-red-600' : 'text-blue-600'
                              }`}>
                                {analysisResult?.verdict?.method_predictions?.spearman_correlation?.toUpperCase() || 'N/A'}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center justify-center">
                            <span className="text-lg">→</span>
                            <div className="mx-3 bg-purple-50 border-2 border-purple-200 rounded-lg px-4 py-2">
                              <div className="text-xs text-purple-600">BreakHis Consensus</div>
                              <div className={`font-bold text-lg ${
                                analysisResult?.verdict?.summary?.breakhis_consensus === 'malignant' ? 'text-red-600' : 'text-blue-600'
                              }`}>
                                {analysisResult?.verdict?.summary?.breakhis_consensus?.toUpperCase() || 'UNKNOWN'}
                              </div>
                              <div className="text-xs text-gray-600">
                                {analysisResult?.verdict?.summary?.agreement_status || ''}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Step 2: BACH Subtype */}
                        <div className="border-t pt-4">
                          <h5 className="font-medium mb-3 flex items-center gap-2">
                            <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-sm">Step 2</span>
                            BACH Subtype Classification
                          </h5>
                          <div className="text-sm text-gray-600 mb-3">
                            {analysisResult?.verdict?.summary?.breakhis_consensus === 'malignant' 
                              ? 'Malignant → Check: Invasive vs In Situ'
                              : 'Benign → Check: Normal vs Benign'
                            }
                          </div>
                          <div className="flex items-center justify-center">
                            <div className="bg-green-50 border-2 border-green-200 rounded-lg px-4 py-2">
                              <div className="text-xs text-green-600">BACH Consensus</div>
                              <div className={`font-bold text-lg ${
                                (analysisResult?.verdict?.summary?.bach_consensus === 'malignant' || analysisResult.verdict.final_prediction === 'malignant') ? 'text-red-600' :
                                (analysisResult?.verdict?.summary?.bach_consensus === 'invasive' || analysisResult.verdict.final_prediction === 'invasive') ? 'text-purple-600' :
                                (analysisResult?.verdict?.summary?.bach_consensus === 'insitu' || analysisResult.verdict.final_prediction === 'insitu') ? 'text-orange-600' :
                                (analysisResult?.verdict?.summary?.bach_consensus === 'normal' || analysisResult.verdict.final_prediction === 'normal') ? 'text-green-600' : 'text-blue-600'
                              }`}>
                                {(analysisResult?.verdict?.summary?.bach_consensus || 
                                  analysisResult?.verdict?.summary?.bach_subtype ||
                                  analysisResult.verdict.final_prediction)?.toUpperCase() || 'UNKNOWN'}
                              </div>
                            </div>
                            <span className="mx-3 text-lg">→</span>
                            <div className="bg-orange-50 border-2 border-orange-200 rounded-lg px-4 py-2">
                              <div className="text-xs text-orange-600">Final Result</div>
                              <div className={`font-bold text-xl ${
                                analysisResult.verdict.final_prediction === 'malignant' ? 'text-red-600' :
                                analysisResult.verdict.final_prediction === 'invasive' ? 'text-purple-600' :
                                analysisResult.verdict.final_prediction === 'insitu' ? 'text-orange-600' :
                                analysisResult.verdict.final_prediction === 'normal' ? 'text-green-600' : 'text-blue-600'
                              }`}>
                                {analysisResult.verdict.final_prediction.toUpperCase()}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Method Breakdown */}
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Prediction Summary */}
                      <div className="bg-white rounded-lg p-6 shadow-sm border">
                        <h4 className="font-semibold mb-3">Method-wise Predictions</h4>
                        <div className="space-y-2">
                          {analysisResult?.verdict?.method_predictions && Object.entries(analysisResult.verdict.method_predictions).map(([method, prediction]) => (
                            <div key={method} className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">{method.replaceAll('_', ' ')}</span>
                              <span className={`font-medium ${
                                prediction === 'malignant' ? 'text-red-600' : 
                                prediction === 'invasive' ? 'text-purple-600' :
                                prediction === 'insitu' ? 'text-orange-600' :
                                prediction === 'normal' ? 'text-green-600' : 'text-blue-600'
                              }`}>
                                {prediction || 'unknown'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Consensus Details */}
                      <div className="bg-white rounded-lg p-6 shadow-sm border">
                        <h4 className="font-semibold mb-3">Consensus Analysis</h4>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">BreakHis consensus</span>
                            <span className={`font-semibold ${
                              analysisResult?.verdict?.summary?.breakhis_consensus === 'malignant' ? 'text-red-600' : 'text-blue-600'
                            }`}>
                              {analysisResult?.verdict?.summary?.breakhis_consensus?.toUpperCase() || 'UNKNOWN'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">BACH consensus</span>
                            <span className={`font-semibold ${
                              (analysisResult?.verdict?.summary?.bach_consensus === 'malignant' || analysisResult.verdict.final_prediction === 'malignant') ? 'text-red-600' :
                              (analysisResult?.verdict?.summary?.bach_consensus === 'invasive' || analysisResult.verdict.final_prediction === 'invasive') ? 'text-purple-600' :
                              (analysisResult?.verdict?.summary?.bach_consensus === 'insitu' || analysisResult.verdict.final_prediction === 'insitu') ? 'text-orange-600' :
                              (analysisResult?.verdict?.summary?.bach_consensus === 'normal' || analysisResult.verdict.final_prediction === 'normal') ? 'text-green-600' : 'text-blue-600'
                            }`}>
                              {(analysisResult?.verdict?.summary?.bach_consensus || 
                                analysisResult?.verdict?.summary?.bach_subtype ||
                                analysisResult.verdict.final_prediction)?.toUpperCase() || 'UNKNOWN'}
                            </span>
                          </div>
                          <div className="border-t pt-3 space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Confidence level</span>
                              <span className={`font-bold ${
                                (analysisResult?.verdict?.summary?.confidence_level === 'HIGH' || 
                                 analysisResult?.verdict?.hierarchical_details?.confidence_level === 'HIGH' ||
                                 (analysisResult?.verdict?.recommendation && analysisResult.verdict.recommendation.includes('HIGH'))) ? 'text-green-600' : 'text-yellow-600'
                              }`}>
                                {analysisResult?.verdict?.summary?.confidence_level || 
                                 analysisResult?.verdict?.hierarchical_details?.confidence_level ||
                                 (analysisResult?.verdict?.recommendation?.includes('HIGH') ? 'HIGH' : 
                                  analysisResult?.verdict?.recommendation?.includes('LOW') ? 'LOW' : 'UNKNOWN')}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Agreement status</span>
                              <span className="font-medium text-gray-700">
                                {analysisResult?.verdict?.summary?.agreement_status || 
                                 analysisResult?.verdict?.hierarchical_details?.agreement_status || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Method</span>
                              <span className="font-medium text-gray-700">
                                {analysisResult?.verdict?.summary?.classification_method || 'Consensus'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Highest similarity</span>
                              <span className="font-medium">{analysisResult?.verdict?.summary?.highest_similarity ? (analysisResult.verdict.summary.highest_similarity * 100).toFixed(1) : '0.0'}%</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Domain-Invariant Pooled Coordinate Predictions */}
                    {analysisResult?.verdict?.coordinate_predictions && (
                      <div className="mt-6">
                        <h4 className="text-lg font-semibold mb-4">🧠 Domain-Invariant Pooled Coordinate Predictions</h4>
                        <p className="text-sm text-gray-600 mb-4">
                          Predictions based on coordinates from the pooled BreakHis + BACH embedding (Domain-invariant tab)
                        </p>
                        <div className="grid md:grid-cols-3 gap-4 mb-6">
                          {['umap', 'tsne', 'pca'].map(method => (
                            <div key={method} className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4 shadow-sm border border-blue-200">
                              <h5 className="font-medium mb-3 text-center text-blue-800">{method.toUpperCase()}</h5>
                              
                              {/* Pooled prediction */}
                              {analysisResult.verdict.coordinate_predictions[method]?.pooled && (
                                <div className="p-3 bg-white rounded border border-blue-200">
                                  <div className="text-sm font-medium text-blue-800 mb-2">
                                    Prediction: <span className={`${
                                      analysisResult.verdict.coordinate_predictions[method].pooled.prediction === 'malignant' 
                                        ? 'text-red-600' : 'text-green-600'
                                    }`}>
                                      {analysisResult.verdict.coordinate_predictions[method].pooled.prediction.toUpperCase()}
                                    </span>
                                  </div>
                                  <div className="text-xs text-gray-600 space-y-1">
                                    <div>Closest: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].pooled.closest_label}</span></div>
                                    <div>Distance: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].pooled.closest_distance.toFixed(3)}</span></div>
                                    <div>Confidence: <span className="font-medium">{(analysisResult.verdict.coordinate_predictions[method].pooled.confidence * 100).toFixed(1)}%</span></div>
                                    {analysisResult.verdict.coordinate_predictions[method].pooled.consensus_votes && (
                                      <div className="text-xs mt-2 pt-2 border-t border-gray-200">
                                        <div>Top 5 Consensus:</div>
                                        <div>Malignant: <span className="font-medium text-red-600">{analysisResult.verdict.coordinate_predictions[method].pooled.consensus_votes.malignant}</span></div>
                                        <div>Benign: <span className="font-medium text-green-600">{analysisResult.verdict.coordinate_predictions[method].pooled.consensus_votes.benign}</span></div>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Coordinate-Based Predictions (Separate Datasets) */}
                    {analysisResult?.verdict?.coordinate_predictions && (
                      <div className="mt-6">
                        <h4 className="text-lg font-semibold mb-4">📊 Dataset-Specific Coordinate Predictions</h4>
                        <p className="text-sm text-gray-600 mb-4">
                          Predictions based on separate BreakHis and BACH coordinates
                        </p>
                        <div className="grid md:grid-cols-3 gap-4">
                          {['umap', 'tsne', 'pca'].map(method => (
                            <div key={method} className="bg-white rounded-lg p-4 shadow-sm border">
                              <h5 className="font-medium mb-3 text-center">{method.toUpperCase()}</h5>
                              
                              {/* BreakHis prediction */}
                              {analysisResult.verdict.coordinate_predictions[method]?.breakhis && (
                                <div className="mb-3 p-3 bg-purple-50 rounded">
                                  <div className="text-sm font-medium text-purple-800 mb-1">BreakHis</div>
                                  <div className="text-xs text-gray-600">
                                    Closest: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].breakhis.closest_label}</span>
                                  </div>
                                  <div className="text-xs text-gray-600">
                                    Distance: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].breakhis.closest_distance.toFixed(3)}</span>
                                  </div>
                                  <div className="text-xs text-gray-600">
                                    Confidence: <span className="font-medium">{(analysisResult.verdict.coordinate_predictions[method].breakhis.confidence * 100).toFixed(1)}%</span>
                                  </div>
                                </div>
                              )}
                              
                              {/* BACH prediction */}
                              {analysisResult.verdict.coordinate_predictions[method]?.bach && (
                                <div className="p-3 bg-green-50 rounded">
                                  <div className="text-sm font-medium text-green-800 mb-1">BACH</div>
                                  <div className="text-xs text-gray-600">
                                    Closest: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].bach.closest_label}</span>
                                  </div>
                                  <div className="text-xs text-gray-600">
                                    Distance: <span className="font-medium">{analysisResult.verdict.coordinate_predictions[method].bach.closest_distance.toFixed(3)}</span>
                                  </div>
                                  <div className="text-xs text-gray-600">
                                    Confidence: <span className="font-medium">{(analysisResult.verdict.coordinate_predictions[method].bach.confidence * 100).toFixed(1)}%</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Similarity-Based Predictions */}
                    {analysisResult?.verdict?.similarity_predictions && (
                      <div className="mt-6">
                        <h4 className="text-lg font-semibold mb-4">Similarity-Based Predictions (L2 Normalized)</h4>
                        <div className="grid md:grid-cols-2 gap-6">
                          
                          {/* BreakHis Similarity */}
                          {analysisResult.verdict.similarity_predictions.breakhis && (
                            <div className="bg-white rounded-lg p-6 shadow-sm border">
                              <h5 className="font-medium mb-4 text-purple-800">BreakHis Dataset</h5>
                              
                              <div className="mb-4 p-3 bg-purple-50 rounded">
                                <div className="text-sm font-medium mb-2">Best Match</div>
                                <div className="text-xs text-gray-600 space-y-1">
                                  <div>Label: <span className="font-medium">{analysisResult.verdict.similarity_predictions.breakhis.best_match.label}</span></div>
                                  <div>Similarity: <span className="font-medium">{(analysisResult.verdict.similarity_predictions.breakhis.best_match.similarity * 100).toFixed(1)}%</span></div>
                                </div>
                              </div>
                              
                              <div className="p-3 bg-gray-50 rounded">
                                <div className="text-sm font-medium mb-2">Consensus (Top 5)</div>
                                <div className="text-xs text-gray-600 space-y-1">
                                  <div>Prediction: <span className="font-medium">{analysisResult.verdict.similarity_predictions.breakhis.consensus.label}</span></div>
                                  <div>Confidence: <span className="font-medium">{(analysisResult.verdict.similarity_predictions.breakhis.consensus.confidence * 100).toFixed(1)}%</span></div>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* BACH Similarity */}
                          {analysisResult.verdict.similarity_predictions.bach && (
                            <div className="bg-white rounded-lg p-6 shadow-sm border">
                              <h5 className="font-medium mb-4 text-green-800">BACH Dataset</h5>
                              
                              <div className="mb-4 p-3 bg-green-50 rounded">
                                <div className="text-sm font-medium mb-2">Best Match</div>
                                <div className="text-xs text-gray-600 space-y-1">
                                  <div>Label: <span className="font-medium">{analysisResult.verdict.similarity_predictions.bach.best_match.label}</span></div>
                                  <div>Similarity: <span className="font-medium">{(analysisResult.verdict.similarity_predictions.bach.best_match.similarity * 100).toFixed(1)}%</span></div>
                                </div>
                              </div>
                              
                              <div className="p-3 bg-gray-50 rounded">
                                <div className="text-sm font-medium mb-2">Consensus (Top 5)</div>
                                <div className="text-xs text-gray-600 space-y-1">
                                  <div>Prediction: <span className="font-medium">{analysisResult.verdict.similarity_predictions.bach.consensus.label}</span></div>
                                  <div>Confidence: <span className="font-medium">{(analysisResult.verdict.similarity_predictions.bach.consensus.confidence * 100).toFixed(1)}%</span></div>
                                </div>
                              </div>
                            </div>
                          )}

                        </div>
                      </div>
                    )}

                    {/* Correlation-Based Predictions (Pearson & Spearman) */}
                    {analysisResult?.verdict?.correlation_predictions && (
                      <div className="mt-8 space-y-6">
                        
                        {/* Pearson Correlation Section */}
                        {analysisResult.verdict.correlation_predictions.pearson && (
                          <div className="bg-amber-50 rounded-lg p-6">
                            <h4 className="font-semibold mb-4 text-center text-amber-700">Pearson Correlation-Based Predictions</h4>
                            <p className="text-sm text-center text-amber-600 mb-4">Linear correlation analysis between feature vectors</p>
                            
                            <div className="grid grid-cols-2 gap-6">
                              {/* BreakHis Pearson Analysis */}
                              {analysisResult.verdict.correlation_predictions.pearson.dataset_predictions?.breakhis && (
                                <div className="bg-white rounded-lg p-4 border border-amber-200">
                                  <h5 className="font-medium text-amber-700 mb-3">BreakHis Analysis</h5>
                                  
                                  <div className="space-y-3">
                                    <div className="bg-amber-25 p-3 rounded border-l-4 border-amber-300">
                                      <div className="text-sm font-medium text-amber-700 mb-1">Best Match</div>
                                      <div>Label: <span className="font-medium">{analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.breakhis.best_match.label}</span></div>
                                      <div>Correlation: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.breakhis.best_match.similarity * 100).toFixed(1)}%</span></div>
                                    </div>
                                    
                                    <div className="bg-amber-25 p-3 rounded border-l-4 border-amber-400">
                                      <div className="text-sm font-medium text-amber-700 mb-1">Consensus (Top 5)</div>
                                      <div>Prediction: <span className="font-medium">{analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.breakhis.consensus.label}</span></div>
                                      <div>Confidence: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.breakhis.consensus.confidence * 100).toFixed(1)}%</span></div>
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* BACH Pearson Analysis */}
                              {analysisResult.verdict.correlation_predictions.pearson.dataset_predictions?.bach && (
                                <div className="bg-white rounded-lg p-4 border border-amber-200">
                                  <h5 className="font-medium text-amber-700 mb-3">BACH Analysis</h5>
                                  
                                  <div className="space-y-3">
                                    <div className="bg-amber-25 p-3 rounded border-l-4 border-amber-300">
                                      <div className="text-sm font-medium text-amber-700 mb-1">Best Match</div>
                                      <div>Label: <span className="font-medium">{analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.bach.best_match.label}</span></div>
                                      <div>Correlation: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.bach.best_match.similarity * 100).toFixed(1)}%</span></div>
                                    </div>
                                    
                                    <div className="bg-amber-25 p-3 rounded border-l-4 border-amber-400">
                                      <div className="text-sm font-medium text-amber-700 mb-1">Consensus (Top 5)</div>
                                      <div>Prediction: <span className="font-medium">{analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.bach.consensus.label}</span></div>
                                      <div>Confidence: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.pearson.dataset_predictions.bach.consensus.confidence * 100).toFixed(1)}%</span></div>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Pearson Overall Stats */}
                            <div className="mt-4 bg-white rounded-lg p-4 border border-amber-200">
                              <h5 className="font-medium text-amber-700 mb-2">Overall Pearson Statistics</h5>
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-amber-600">Highest Correlation:</span> 
                                  <span className="font-medium ml-2">{(analysisResult.verdict.correlation_predictions.pearson.overall_top_similarity * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                  <span className="text-amber-600">Method:</span> 
                                  <span className="font-medium ml-2 capitalize">{analysisResult.verdict.correlation_predictions.pearson.method}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Spearman Correlation Section */}
                        {analysisResult.verdict.correlation_predictions.spearman && (
                          <div className="bg-purple-50 rounded-lg p-6">
                            <h4 className="font-semibold mb-4 text-center text-purple-700">Spearman Correlation-Based Predictions</h4>
                            <p className="text-sm text-center text-purple-600 mb-4">Rank-based correlation analysis (non-parametric)</p>
                            
                            <div className="grid grid-cols-2 gap-6">
                              {/* BreakHis Spearman Analysis */}
                              {analysisResult.verdict.correlation_predictions.spearman.dataset_predictions?.breakhis && (
                                <div className="bg-white rounded-lg p-4 border border-purple-200">
                                  <h5 className="font-medium text-purple-700 mb-3">BreakHis Analysis</h5>
                                  
                                  <div className="space-y-3">
                                    <div className="bg-purple-25 p-3 rounded border-l-4 border-purple-300">
                                      <div className="text-sm font-medium text-purple-700 mb-1">Best Match</div>
                                      <div>Label: <span className="font-medium">{analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.breakhis.best_match.label}</span></div>
                                      <div>Correlation: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.breakhis.best_match.similarity * 100).toFixed(1)}%</span></div>
                                    </div>
                                    
                                    <div className="bg-purple-25 p-3 rounded border-l-4 border-purple-400">
                                      <div className="text-sm font-medium text-purple-700 mb-1">Consensus (Top 5)</div>
                                      <div>Prediction: <span className="font-medium">{analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.breakhis.consensus.label}</span></div>
                                      <div>Confidence: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.breakhis.consensus.confidence * 100).toFixed(1)}%</span></div>
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* BACH Spearman Analysis */}
                              {analysisResult.verdict.correlation_predictions.spearman.dataset_predictions?.bach && (
                                <div className="bg-white rounded-lg p-4 border border-purple-200">
                                  <h5 className="font-medium text-purple-700 mb-3">BACH Analysis</h5>
                                  
                                  <div className="space-y-3">
                                    <div className="bg-purple-25 p-3 rounded border-l-4 border-purple-300">
                                      <div className="text-sm font-medium text-purple-700 mb-1">Best Match</div>
                                      <div>Label: <span className="font-medium">{analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.bach.best_match.label}</span></div>
                                      <div>Correlation: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.bach.best_match.similarity * 100).toFixed(1)}%</span></div>
                                    </div>
                                    
                                    <div className="bg-purple-25 p-3 rounded border-l-4 border-purple-400">
                                      <div className="text-sm font-medium text-purple-700 mb-1">Consensus (Top 5)</div>
                                      <div>Prediction: <span className="font-medium">{analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.bach.consensus.label}</span></div>
                                      <div>Confidence: <span className="font-medium">{(analysisResult.verdict.correlation_predictions.spearman.dataset_predictions.bach.consensus.confidence * 100).toFixed(1)}%</span></div>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Spearman Overall Stats */}
                            <div className="mt-4 bg-white rounded-lg p-4 border border-purple-200">
                              <h5 className="font-medium text-purple-700 mb-2">Overall Spearman Statistics</h5>
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-purple-600">Highest Correlation:</span> 
                                  <span className="font-medium ml-2">{(analysisResult.verdict.correlation_predictions.spearman.overall_top_similarity * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                  <span className="text-purple-600">Method:</span> 
                                  <span className="font-medium ml-2 capitalize">{analysisResult.verdict.correlation_predictions.spearman.method}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Correlation Comparison Summary */}
                        <div className="bg-gray-50 rounded-lg p-6">
                          <h4 className="font-semibold mb-4 text-center text-gray-700">Correlation Methods Comparison</h4>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div className="text-center">
                              <div className="font-medium text-blue-700">Cosine Similarity</div>
                              <div className="text-blue-600">Geometric angle between vectors</div>
                            </div>
                            <div className="text-center">
                              <div className="font-medium text-amber-700">Pearson Correlation</div>
                              <div className="text-amber-600">Linear relationship strength</div>
                            </div>
                            <div className="text-center">
                              <div className="font-medium text-purple-700">Spearman Correlation</div>
                              <div className="text-purple-600">Rank-based relationship</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}