'use client'

import { Brain, Target, AlertCircle } from 'lucide-react'
import { StageResults } from './StageResults'

interface TieredPredictionProps {
  tieredPrediction?: {
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
}

export function TieredPredictionDisplay({ tieredPrediction }: TieredPredictionProps) {
  if (!tieredPrediction) {
    return (
      <div className="bg-red-50 rounded-xl p-6 border-2 border-red-200">
        <div className="text-center text-red-600">
          <AlertCircle className="w-8 h-8 mx-auto mb-2" />
          <p className="font-medium">Tiered prediction system not available</p>
          <p className="text-sm">Please ensure the backend API is running</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Final Prediction Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 mb-6 text-white">
        <div className="text-center">
          <div className="text-3xl font-bold mb-2">
            {tieredPrediction.tiered_final_prediction?.toUpperCase()}
          </div>
          <div className="text-blue-100 text-lg">
            🏥 {tieredPrediction.clinical_pathway}
          </div>
          <div className="text-blue-200 text-sm mt-2">
            Status: {tieredPrediction.system_status?.replace('_', ' ').toUpperCase()}
          </div>
        </div>
      </div>

      {/* Two-Stage Analysis */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Stage 1: BreakHis */}
        <StageResults
          title="Stage 1: BreakHis Binary"
          subtitle="Malignant vs Benign Classification"
          consensus={tieredPrediction.stage_1_breakhis.consensus}
          voteBreakdown={tieredPrediction.stage_1_breakhis.vote_breakdown}
          totalClassifiers={tieredPrediction.stage_1_breakhis.total_classifiers}
          classifiers={tieredPrediction.stage_1_breakhis.classifiers}
          colorScheme="green"
          icon={<Target className="w-5 h-5 text-white" />}
        />

        {/* Stage 2: Specialized BACH */}
        {tieredPrediction.stage_2_bach_specialized ? (
          <StageResults
            title={`Stage 2: ${tieredPrediction.stage_2_bach_specialized.task}`}
            subtitle="Specialized Binary Classification"
            consensus={tieredPrediction.stage_2_bach_specialized.consensus}
            voteBreakdown={tieredPrediction.stage_2_bach_specialized.vote_breakdown}
            totalClassifiers={tieredPrediction.stage_2_bach_specialized.total_classifiers}
            classifiers={tieredPrediction.stage_2_bach_specialized.classifiers}
            colorScheme="purple"
            icon={<Brain className="w-5 h-5 text-white" />}
          />
        ) : (
          <div className="bg-gray-50 rounded-xl p-6 border-2 border-gray-200">
            <div className="text-center text-gray-500">
              <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Stage 2 classifier not deployed</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}