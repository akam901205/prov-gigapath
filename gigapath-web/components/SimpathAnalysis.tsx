'use client'

import { useState } from 'react'
import { Search, Target, BarChart3, TrendingUp } from 'lucide-react'
import axios from 'axios'

interface SimilarityResult {
  metric: string
  breakhis_best_match: {
    filename: string
    label: string
    score: number
    rank: number
  }
  bach_best_match: {
    filename: string
    label: string
    score: number
    rank: number
  }
}

interface SimpathResult {
  similarity_analysis: SimilarityResult[]
  summary: {
    breakhis_consensus: string
    bach_consensus: string
    most_reliable_metric: string
    confidence_score: number
  }
}

interface SimpathAnalysisProps {
  imageFile?: File
  imagePreview?: string
}

export default function SimpathAnalysis({ imageFile, imagePreview }: SimpathAnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<SimpathResult | null>(null)
  const [error, setError] = useState<string>('')
  const [currentStep, setCurrentStep] = useState('')

  const runSimpathAnalysis = async () => {
    if (!imageFile) {
      setError('Please select an image first')
      return
    }

    setIsAnalyzing(true)
    setError('')
    setResult(null)
    setCurrentStep('Computing similarity metrics against 2,217 BACH samples...')

    try {
      // Debug: Check if FormData creation fails
      console.log('Creating FormData for file:', imageFile.name, 'size:', imageFile.size)
      
      let formData: FormData
      try {
        formData = new FormData()
        console.log('FormData created successfully')
        formData.append('image', imageFile)
        console.log('Image appended to FormData successfully')
      } catch (formDataError) {
        console.error('FormData creation failed:', formDataError)
        setError(`FormData error: ${(formDataError as Error).message}`)
        setIsAnalyzing(false)
        return
      }
      
      const response = await axios.post('/api/simpath-analysis', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 900000, // 15 minutes like true-tiered
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      })
      
      // Handle 202 processing status
      if (response.status === 202) {
        setCurrentStep('Analysis started! Processing against 2,217 samples...')
        setError(`Processing Status: ${response.data.message}`)
        setIsAnalyzing(false)
        return
      }

      if (response.data.status === 'success') {
        setResult(response.data)
        setCurrentStep('Similarity analysis complete!')
      } else {
        setError(response.data.error || 'Analysis failed')
      }

    } catch (err: any) {
      console.error('Simpath Analysis error:', err)
      setError(`Analysis failed: ${err.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getMetricColor = (metric: string) => {
    const colors = {
      'cosine': 'bg-blue-100 text-blue-800',
      'euclidean': 'bg-green-100 text-green-800', 
      'manhattan': 'bg-purple-100 text-purple-800',
      'chebyshev': 'bg-yellow-100 text-yellow-800',
      'braycurtis': 'bg-pink-100 text-pink-800',
      'canberra': 'bg-indigo-100 text-indigo-800',
      'seuclidean': 'bg-red-100 text-red-800',
      'pearson': 'bg-orange-100 text-orange-800',
      'spearman': 'bg-teal-100 text-teal-800',
      'dcor': 'bg-gray-100 text-gray-800'
    }
    return colors[metric] || 'bg-gray-100 text-gray-800'
  }

  const getLabelColor = (label: string) => {
    return label === 'malignant' || label === 'invasive' || label === 'insitu' 
      ? 'text-red-600 bg-red-50 border-red-200' 
      : 'text-green-600 bg-green-50 border-green-200'
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-teal-500 to-cyan-500 text-white p-6 rounded-lg">
        <div className="flex items-center gap-3 mb-4">
          <Search className="h-8 w-8" />
          <div>
            <h2 className="text-2xl font-bold">Simpath Analysis</h2>
            <p className="text-teal-100">Multi-metric similarity search across BreakHis & BACH datasets</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
          {['Cosine', 'Euclidean', 'Manhattan', 'Chebyshev', 'Bray-Curtis'].map(metric => (
            <div key={metric} className="bg-white/10 p-2 rounded text-center">
              <div className="font-semibold">{metric}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Upload Section */}
      {!imageFile && (
        <div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-lg">
          <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">Upload an image in the main upload area to run Simpath analysis</p>
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
                onClick={runSimpathAnalysis}
                disabled={isAnalyzing}
                className="mt-3 bg-teal-600 text-white px-6 py-2 rounded-lg hover:bg-teal-700 disabled:opacity-50 flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Run Simpath Analysis
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isAnalyzing && (
        <div className="bg-teal-50 border border-teal-200 p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-teal-600 border-t-transparent"></div>
            <h3 className="text-lg font-semibold text-teal-800">Simpath Analysis in Progress</h3>
          </div>
          <p className="text-teal-700 mb-2">{currentStep}</p>
          <div className="bg-teal-100 rounded-full h-2 overflow-hidden">
            <div className="bg-teal-600 h-full animate-pulse" style={{ width: '70%' }}></div>
          </div>
          <p className="text-sm text-teal-600 mt-2">Computing 10 similarity metrics across 2,217 samples...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
          <div className="flex items-center gap-2 text-red-800">
            <Target className="h-5 w-5" />
            <span className="font-medium">Analysis Failed</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Similarity Analysis Summary
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <h4 className="font-semibold text-purple-800 mb-2">🔬 BreakHis Consensus</h4>
                <div className={`inline-block px-3 py-1 rounded-full border ${getLabelColor(result.summary.breakhis_consensus)}`}>
                  <span className="font-medium">{result.summary.breakhis_consensus.toUpperCase()}</span>
                </div>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-800 mb-2">🧬 BACH Consensus</h4>
                <div className={`inline-block px-3 py-1 rounded-full border ${getLabelColor(result.summary.bach_consensus)}`}>
                  <span className="font-medium">{result.summary.bach_consensus.toUpperCase()}</span>
                </div>
              </div>
            </div>
            
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-600">
                Most Reliable Metric: <span className="font-medium">{result.summary.most_reliable_metric}</span>
                | Confidence: <span className="font-medium">{(result.summary.confidence_score * 100).toFixed(1)}%</span>
              </p>
            </div>
          </div>

          {/* BreakHis Section */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Target className="h-5 w-5 text-purple-600" />
              🔬 BreakHis Best Matches by Metric
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="text-left p-3 font-medium">Similarity Metric</th>
                    <th className="text-left p-3 font-medium">Best Match</th>
                    <th className="text-left p-3 font-medium">Label</th>
                    <th className="text-left p-3 font-medium">Score</th>
                    <th className="text-left p-3 font-medium">Rank</th>
                  </tr>
                </thead>
                <tbody>
                  {result.similarity_analysis.map((item, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getMetricColor(item.metric)}`}>
                          {item.metric.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-3 font-mono text-xs">{item.breakhis_best_match.filename}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded border text-xs font-medium ${getLabelColor(item.breakhis_best_match.label)}`}>
                          {item.breakhis_best_match.label}
                        </span>
                      </td>
                      <td className="p-3 font-mono">{item.breakhis_best_match.score.toFixed(4)}</td>
                      <td className="p-3">#{item.breakhis_best_match.rank}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* BACH Section */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-green-600" />
              🧬 BACH Best Matches by Metric
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="text-left p-3 font-medium">Similarity Metric</th>
                    <th className="text-left p-3 font-medium">Best Match</th>
                    <th className="text-left p-3 font-medium">Label</th>
                    <th className="text-left p-3 font-medium">Score</th>
                    <th className="text-left p-3 font-medium">Rank</th>
                  </tr>
                </thead>
                <tbody>
                  {result.similarity_analysis.map((item, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getMetricColor(item.metric)}`}>
                          {item.metric.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-3 font-mono text-xs">{item.bach_best_match.filename}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded border text-xs font-medium ${getLabelColor(item.bach_best_match.label)}`}>
                          {item.bach_best_match.label}
                        </span>
                      </td>
                      <td className="p-3 font-mono">{item.bach_best_match.score.toFixed(4)}</td>
                      <td className="p-3">#{item.bach_best_match.rank}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Metric Comparison */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">📊 Cross-Metric Analysis</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3 text-purple-700">BreakHis Label Distribution</h4>
                <div className="space-y-2">
                  {Object.entries(
                    result.similarity_analysis.reduce((acc, item) => {
                      const label = item.breakhis_best_match.label
                      acc[label] = (acc[label] || 0) + 1
                      return acc
                    }, {} as Record<string, number>)
                  ).map(([label, count]) => (
                    <div key={label} className="flex justify-between items-center">
                      <span className={`px-2 py-1 rounded border text-xs ${getLabelColor(label)}`}>
                        {label}
                      </span>
                      <span className="font-medium">{count}/10 metrics</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3 text-green-700">BACH Label Distribution</h4>
                <div className="space-y-2">
                  {Object.entries(
                    result.similarity_analysis.reduce((acc, item) => {
                      const label = item.bach_best_match.label
                      acc[label] = (acc[label] || 0) + 1
                      return acc
                    }, {} as Record<string, number>)
                  ).map(([label, count]) => (
                    <div key={label} className="flex justify-between items-center">
                      <span className={`px-2 py-1 rounded border text-xs ${getLabelColor(label)}`}>
                        {label}
                      </span>
                      <span className="font-medium">{count}/10 metrics</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}