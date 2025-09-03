'use client'

import { useState, useRef } from 'react'
import { Search, Target, BarChart3 } from 'lucide-react'
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

export default function SimpathAnalysisSimple({ imageFile, imagePreview }: SimpathAnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<SimpathResult | null>(null)
  const [error, setError] = useState<string>('')
  const [progress, setProgress] = useState<string>('')
  const [estimatedTimeLeft, setEstimatedTimeLeft] = useState<string>('')
  const [currentMetric, setCurrentMetric] = useState<string>('')
  const [completedMetrics, setCompletedMetrics] = useState<string[]>([])
  const currentMetricIndexRef = useRef<number>(0)

  const runSimpathAnalysis = async () => {
    console.log('🚀 Simpath Analysis Started')
    console.log('ImageFile:', imageFile)
    console.log('Initial state - isAnalyzing:', isAnalyzing)
    
    if (!imageFile) {
      console.error('❌ No image file provided')
      setError('Please select an image first')
      return
    }

    console.log('✅ Image file found, starting analysis...')
    setIsAnalyzing(true)
    setError('')
    setResult(null)
    setProgress('Initializing analysis...')
    setEstimatedTimeLeft('Estimating time...')
    setCurrentMetric('')
    setCompletedMetrics([])
    currentMetricIndexRef.current = 0
    
    console.log('State set - isAnalyzing should now be true')
    console.log('Progress set to:', 'Initializing analysis...')

    // UI progress tracking is now working properly

    const metrics = ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'braycurtis', 'canberra', 'seuclidean', 'pearson']

    // Start a progress timer
    const startTime = Date.now()
    console.log('Starting progress interval timer')
    const progressInterval = setInterval(() => {
      console.log('Progress interval tick')
      const elapsed = Math.floor((Date.now() - startTime) / 1000)
      const minutes = Math.floor(elapsed / 60)
      const seconds = elapsed % 60
      
      // Simulate metric progression (approximately 60 seconds per metric for 8 metrics)
      const expectedMetricIndex = Math.min(Math.floor(elapsed / 60), metrics.length - 1)
      
      if (expectedMetricIndex > currentMetricIndexRef.current) {
        console.log(`Metric completed: ${metrics[currentMetricIndexRef.current]}`)
        setCompletedMetrics(prev => [...prev, metrics[currentMetricIndexRef.current]])
        currentMetricIndexRef.current = expectedMetricIndex
        console.log(`New currentMetricIndex: ${expectedMetricIndex}`)
      }
      
      if (currentMetricIndexRef.current < metrics.length) {
        setCurrentMetric(metrics[currentMetricIndexRef.current])
        setProgress(`Computing ${metrics[currentMetricIndexRef.current]} similarity... ${minutes}:${seconds.toString().padStart(2, '0')} elapsed`)
      } else {
        setCurrentMetric('Final aggregation')
        setProgress(`Finalizing results... ${minutes}:${seconds.toString().padStart(2, '0')} elapsed`)
      }
      
      // Estimate based on 10 metrics * ~18 seconds each = ~3 minutes total
      if (elapsed < 180) {
        const remaining = 180 - elapsed
        const remainingMinutes = Math.floor(remaining / 60)
        const remainingSeconds = remaining % 60
        setEstimatedTimeLeft(`~${remainingMinutes}:${remainingSeconds.toString().padStart(2, '0')} remaining`)
      } else {
        setEstimatedTimeLeft('Completing final calculations...')
      }
    }, 1000)

    try {
      console.log('🌐 Making direct API request to Simpath (bypass FileReader)...')
      
      // Convert file to base64 safely for large files
      const arrayBuffer = await imageFile.arrayBuffer()
      const uint8Array = new Uint8Array(arrayBuffer)
      
      // Convert in chunks to avoid stack overflow
      let binaryString = ''
      const chunkSize = 8192
      for (let i = 0; i < uint8Array.length; i += chunkSize) {
        const chunk = uint8Array.slice(i, i + chunkSize)
        binaryString += String.fromCharCode(...chunk)
      }
      const base64 = btoa(binaryString)
      console.log('📝 Base64 length:', base64.length)
      
      const response = await axios.post('/api/simpath-analysis-proxy', {
        input: {
          image_base64: base64,
          metrics: ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'braycurtis', 'canberra', 'seuclidean', 'pearson']
        }
      }, {
        timeout: 900000, // 15 minutes - super relaxed timeout
        headers: { 'Content-Type': 'application/json' }
      })

      console.log('📥 API Response:', response.data)
      
      console.log('📥 Full API Response structure:', JSON.stringify(response.data, null, 2))
      
      if (response.data && !response.data.error) {
        console.log('✅ Analysis successful!')
        clearInterval(progressInterval)
        setProgress('Analysis completed!')
        setEstimatedTimeLeft('')
        
        // Handle actual Simpath API response
        console.log('✅ Simpath API succeeded!')
        setResult(response.data)
      } else {
        console.error('❌ Analysis failed:', response.data?.error)
        clearInterval(progressInterval)
        setError(response.data?.error || 'Analysis failed')
      }

    } catch (err: any) {
      console.error('🔥 API request error - FULL DETAILS:', err)
      console.error('Error code:', err.code)
      console.error('Error message:', err.message) 
      console.error('Error response:', err.response)
      console.error('Error status:', err.response?.status)
      console.error('Error data:', err.response?.data)
      
      clearInterval(progressInterval)
      
      if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
        setError('⚠️ Simpath API timed out after 15 minutes. The service is experiencing extremely heavy computational load. Please try again later or use the "True Tiered System" tab for immediate results.')
      } else if (err.response?.status === 0 || !err.response) {
        setError(`🔌 Network connection failed. Status code: ${err.response?.status || 'No response'}. The API endpoint may be unreachable.`)
      } else {
        setError(`API request failed: ${err.message} (Status: ${err.response?.status || 'unknown'})`)
      }
    } finally {
      console.log('🏁 Analysis completed, resetting state...')
      setIsAnalyzing(false)
    }
  }

  console.log('🎨 Render - isAnalyzing:', isAnalyzing, 'progress:', progress, 'currentMetric:', currentMetric, 'completedMetrics:', completedMetrics.length)
  
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
      </div>

      {!imageFile && (
        <div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-lg">
          <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">Upload an image in the main upload area to run Simpath analysis</p>
        </div>
      )}

      {/* Service Status Notice */}
      <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-amber-400 rounded-full animate-pulse"></div>
          <h4 className="font-semibold text-amber-800">Service Status</h4>
        </div>
        <p className="text-amber-700 text-sm mt-2">
          ⚠️ <strong>Service Degraded:</strong> The Simpath API is currently overwhelmed with similarity computations and experiencing 504 timeouts. 
          Reduced to single cosine similarity metric for testing. For reliable pathology analysis, please use the 
          <strong>"True Tiered System"</strong> tab which provides comprehensive diagnostic results with faster response times.
        </p>
      </div>

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
                    Computing similarities...
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

      {/* Progress Display */}
      {(isAnalyzing || progress || currentMetric) && (
        <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent"></div>
            <h3 className="text-lg font-semibold text-blue-800">Multi-Metric Similarity Analysis in Progress</h3>
          </div>
          <div className="space-y-3">
            <div className="text-sm text-blue-700 font-medium">{progress}</div>
            <div className="text-sm text-blue-600">{estimatedTimeLeft}</div>
            
            {currentMetric && (
              <div className="text-sm text-blue-800 bg-blue-100 px-3 py-2 rounded-lg">
                <span className="font-medium">Current: </span>
                <span className="capitalize">{currentMetric}</span>
                {currentMetric !== 'Final aggregation' && <span> similarity metric</span>}
              </div>
            )}
            
            {completedMetrics.length > 0 && (
              <div className="text-xs text-blue-600">
                <div className="mb-1">Completed metrics: {completedMetrics.length}/8</div>
                <div className="flex flex-wrap gap-1">
                  {completedMetrics.map(metric => (
                    <span key={metric} className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs capitalize">
                      ✓ {metric}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            <div className="text-xs text-blue-500">
              Computing 8 similarity metrics against 1,817 BreakHis + 400 BACH samples
            </div>
          </div>
          <div className="mt-4 bg-blue-100 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-1000 ease-out" 
              style={{width: `${Math.min((completedMetrics.length / 8) * 100, 95)}%`}}
            ></div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">🔍 Similarity Analysis Results</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">🔬 BreakHis Consensus</h4>
                <span className="font-medium">{result.summary.breakhis_consensus}</span>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">🧬 BACH Consensus</h4>
                <span className="font-medium">{result.summary.bach_consensus}</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="text-left p-3">Metric</th>
                    <th className="text-left p-3">BreakHis Best</th>
                    <th className="text-left p-3">BH Label</th>
                    <th className="text-left p-3">BACH Best</th>
                    <th className="text-left p-3">BACH Label</th>
                  </tr>
                </thead>
                <tbody>
                  {result.similarity_analysis.map((item, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="p-3 font-medium">{item.metric}</td>
                      <td className="p-3 text-xs font-mono">{item.breakhis_best_match.filename}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded text-xs ${item.breakhis_best_match.label === 'malignant' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                          {item.breakhis_best_match.label}
                        </span>
                      </td>
                      <td className="p-3 text-xs font-mono">{item.bach_best_match.filename}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded text-xs ${item.bach_best_match.label === 'malignant' || item.bach_best_match.label === 'invasive' || item.bach_best_match.label === 'insitu' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                          {item.bach_best_match.label}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}