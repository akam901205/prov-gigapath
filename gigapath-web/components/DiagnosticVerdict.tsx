'use client'

import { CheckCircle, AlertCircle, XCircle } from 'lucide-react'

interface DiagnosticVerdictProps {
  analysisResult?: {
    verdict: {
      final_prediction: string
      confidence: number
      recommendation: string
      summary?: {
        confidence_level: string
        agreement_status: string
        classification_method: string
        breakhis_consensus: string
        bach_consensus: string
      }
      hierarchical_details?: {
        confidence_level: string
      }
      method_predictions?: {
        similarity_consensus: string
        pearson_correlation: string
        spearman_correlation: string
        ensemble_final: string
      }
      vote_breakdown?: {
        malignant_votes: number
        benign_votes: number
      }
    }
  }
}

function getConfidenceIcon(confidence: number) {
  if (confidence > 0.8) {
    return <CheckCircle className="w-6 h-6 text-green-600" />
  } else if (confidence > 0.6) {
    return <AlertCircle className="w-6 h-6 text-yellow-600" />
  } else {
    return <XCircle className="w-6 h-6 text-red-600" />
  }
}

export function DiagnosticVerdict({ analysisResult }: DiagnosticVerdictProps) {
  if (!analysisResult?.verdict) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <p className="text-gray-600">No diagnostic verdict available</p>
      </div>
    )
  }

  const { verdict } = analysisResult

  return (
    <div>
      {/* Final Prediction */}
      <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-xl font-bold text-gray-800 mb-2">Final Prediction</h4>
            <div className="flex items-center gap-3">
              {getConfidenceIcon(verdict.confidence)}
              <span className={`text-3xl font-bold ${
                verdict.final_prediction === 'malignant' ? 'text-red-600' : 
                verdict.final_prediction === 'invasive' ? 'text-purple-600' :
                verdict.final_prediction === 'insitu' ? 'text-orange-600' :
                verdict.final_prediction === 'normal' ? 'text-green-600' : 'text-blue-600'
              }`}>
                {verdict.final_prediction.toUpperCase()}
              </span>
              <div className="flex flex-col">
                <span className={`text-lg font-semibold ${
                  (verdict?.summary?.confidence_level === 'HIGH' || 
                   verdict?.hierarchical_details?.confidence_level === 'HIGH' ||
                   (verdict?.recommendation && verdict.recommendation.includes('HIGH'))) ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {verdict?.summary?.confidence_level || 
                   verdict?.hierarchical_details?.confidence_level ||
                   (verdict?.recommendation?.includes('HIGH') ? 'HIGH' : 
                    verdict?.recommendation?.includes('LOW') ? 'LOW' : 'UNKNOWN')} CONFIDENCE
                </span>
                <span className="text-sm text-gray-600">
                  ({(verdict.confidence * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600">Decision Rule</div>
            <div className="text-sm font-medium text-gray-800 max-w-md">
              {verdict?.recommendation || 'Consensus-based classification'}
            </div>
          </div>
        </div>
      </div>

      {/* Consensus Decision Flow */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold mb-4">🎯 Consensus Decision Flow</h4>
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* BreakHis Consensus */}
            <div className="bg-blue-50 rounded-lg p-4">
              <h5 className="font-medium text-blue-700 mb-2">BreakHis Dataset Consensus</h5>
              <div className="text-sm space-y-1">
                <div>Classification: <span className="font-medium">{verdict.summary?.breakhis_consensus || 'N/A'}</span></div>
                <div className="text-xs text-gray-600">Binary malignancy assessment</div>
              </div>
            </div>

            {/* BACH Consensus */}
            <div className="bg-purple-50 rounded-lg p-4">
              <h5 className="font-medium text-purple-700 mb-2">BACH Dataset Consensus</h5>
              <div className="text-sm space-y-1">
                <div>Subtype: <span className="font-medium">{verdict.summary?.bach_consensus || 'N/A'}</span></div>
                <div className="text-xs text-gray-600">Tissue architecture classification</div>
              </div>
            </div>
          </div>

          {/* Method Predictions */}
          {verdict.method_predictions && (
            <div className="mt-4 bg-gray-50 rounded-lg p-4">
              <h5 className="font-medium mb-3">Method Predictions</h5>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Similarity:</span>
                  <div className="font-medium">{verdict.method_predictions.similarity_consensus}</div>
                </div>
                <div>
                  <span className="text-gray-600">Pearson:</span>
                  <div className="font-medium">{verdict.method_predictions.pearson_correlation}</div>
                </div>
                <div>
                  <span className="text-gray-600">Spearman:</span>
                  <div className="font-medium">{verdict.method_predictions.spearman_correlation}</div>
                </div>
                <div>
                  <span className="text-gray-600">Ensemble:</span>
                  <div className="font-medium">{verdict.method_predictions.ensemble_final}</div>
                </div>
              </div>
            </div>
          )}

          {/* Vote Breakdown */}
          {verdict.vote_breakdown && (
            <div className="mt-4 bg-gray-50 rounded-lg p-4">
              <h5 className="font-medium mb-3">Vote Breakdown</h5>
              <div className="flex justify-center gap-8">
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">{verdict.vote_breakdown.malignant_votes}</div>
                  <div className="text-sm text-gray-600">Malignant</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{verdict.vote_breakdown.benign_votes}</div>
                  <div className="text-sm text-gray-600">Benign</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Classification Summary */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h4 className="text-lg font-semibold mb-4">📊 Classification Summary</h4>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Agreement Status:</span>
            <span className={`font-medium ${
              verdict.summary?.agreement_status === 'STRONG' ? 'text-green-600' : 
              verdict.summary?.agreement_status === 'MODERATE' ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {verdict.summary?.agreement_status || 'Unknown'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Classification Method:</span>
            <span className="font-medium text-gray-800 text-right max-w-md">
              {verdict.summary?.classification_method || 'Ensemble classification'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Confidence Level:</span>
            <span className={`font-medium ${
              verdict.summary?.confidence_level === 'HIGH' ? 'text-green-600' : 
              verdict.summary?.confidence_level === 'MODERATE' ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {verdict.summary?.confidence_level || 'Unknown'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}