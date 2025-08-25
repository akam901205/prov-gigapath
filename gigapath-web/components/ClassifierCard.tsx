'use client'

interface ClassifierCardProps {
  algorithm: string
  result: {
    predicted_class: string
    confidence: number
    algorithm?: string
  }
  colorScheme: 'green' | 'purple'
}

export function ClassifierCard({ algorithm, result, colorScheme }: ClassifierCardProps) {
  const getClassColor = (predictedClass: string, scheme: string) => {
    if (scheme === 'green') {
      return predictedClass === 'malignant' ? 'text-red-600' : 'text-green-600'
    }
    return 'text-purple-600'
  }

  return (
    <div className="flex justify-between text-sm">
      <span className="capitalize">{algorithm.replace('_', ' ')}</span>
      <span className={`font-medium ${getClassColor(result.predicted_class, colorScheme)}`}>
        {result.predicted_class} ({(result.confidence * 100).toFixed(1)}%)
      </span>
    </div>
  )
}