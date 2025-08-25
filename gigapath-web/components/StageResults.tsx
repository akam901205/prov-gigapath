'use client'

import { ReactNode } from 'react'
import { ClassifierCard } from './ClassifierCard'

interface StageResultsProps {
  title: string
  subtitle: string
  consensus: string
  voteBreakdown: any
  totalClassifiers: number
  classifiers: {
    logistic_regression: any
    svm_rbf: any
    xgboost: any
  }
  colorScheme: 'green' | 'purple'
  icon: ReactNode
}

export function StageResults({ 
  title, 
  subtitle, 
  consensus, 
  voteBreakdown, 
  totalClassifiers, 
  classifiers, 
  colorScheme,
  icon 
}: StageResultsProps) {
  const colorClasses = {
    green: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      iconBg: 'bg-green-600',
      text: 'text-green-600',
      consensusColor: consensus === 'malignant' ? 'text-red-600' : 'text-green-600'
    },
    purple: {
      bg: 'bg-purple-50',
      border: 'border-purple-200', 
      iconBg: 'bg-purple-600',
      text: 'text-purple-600',
      consensusColor: 'text-purple-600'
    }
  }

  const colors = colorClasses[colorScheme]

  return (
    <div className={`${colors.bg} rounded-xl p-6 border-2 ${colors.border}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className={`${colors.iconBg} rounded-full p-2`}>
          {icon}
        </div>
        <div>
          <h4 className="text-lg font-bold text-gray-900">{title}</h4>
          <p className={`${colors.text} text-sm`}>{subtitle}</p>
        </div>
      </div>
      
      <div className="bg-white rounded-lg p-4">
        <div className="text-center mb-3">
          <div className={`text-2xl font-bold ${colors.consensusColor}`}>
            {consensus?.toUpperCase()}
          </div>
          <div className="text-sm text-gray-600">
            {colorScheme === 'green' ? (
              `${voteBreakdown.malignant}/${totalClassifiers} malignant votes`
            ) : (
              `Consensus from ${totalClassifiers} specialized classifiers`
            )}
          </div>
        </div>
        
        <div className="space-y-2">
          {Object.entries(classifiers).map(([alg, result]: [string, any]) => {
            if (!result) return null
            return (
              <ClassifierCard
                key={alg}
                algorithm={alg}
                result={result}
                colorScheme={colorScheme}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}