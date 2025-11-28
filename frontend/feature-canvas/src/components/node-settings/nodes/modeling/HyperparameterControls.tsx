import React from 'react';
import type { ModelHyperparameterField, HyperparameterTuningJobSummary } from '../../../../api';
import type { HyperparamPreset } from './BestHyperparamsModal';

type HyperparameterControlsProps = {
	modelType: string | null;
	showAdvanced: boolean;
	hyperparamFields: ModelHyperparameterField[];
	hyperparamValues: Record<string, any>;
	onHyperparamChange: (name: string, value: any) => void;
	onToggleAdvanced: () => void;
	isLoading: boolean;
	error: Error | null;
	primaryPreset: HyperparamPreset | null;
	latestTuningJob: HyperparameterTuningJobSummary | null;
	applyStatus: { message: string; tone: 'info' | 'warning' | 'success' } | null;
	onApplyBestParams: () => void;
	onBrowsePresets: () => void;
	applyButtonDisabled: boolean;
	browsePresetsDisabled: boolean;
	latestTuningRunNumber: number | null;
	latestTuningRelative: string | null;
	primaryPresetRelative: string | null;
	primaryPresetScoreLabel: string | null;
	applyStatusColor: string;
	renderParameterField: (parameter: any) => React.ReactNode;
	hyperparametersParameter: any;
};

export const HyperparameterControls: React.FC<HyperparameterControlsProps> = ({
	modelType,
	showAdvanced,
	hyperparamFields,
	hyperparamValues,
	onHyperparamChange,
	onToggleAdvanced,
	isLoading,
	error,
	primaryPreset,
	latestTuningJob,
	applyStatus,
	onApplyBestParams,
	onBrowsePresets,
	applyButtonDisabled,
	browsePresetsDisabled,
	latestTuningRunNumber,
	latestTuningRelative,
	primaryPresetRelative,
	primaryPresetScoreLabel,
	applyStatusColor,
	renderParameterField,
	hyperparametersParameter,
}) => {
	const renderHyperparamField = (field: ModelHyperparameterField) => {
		const currentValue = hyperparamValues[field.name] ?? field.default;
		const fieldId = `hyperparam-${field.name}`;

		return (
			<div key={field.name} className="canvas-modal__parameter">
				<label htmlFor={fieldId} className="canvas-modal__parameter-label">
					{field.label}
					{field.description && (
						<span className="canvas-modal__parameter-description">{field.description}</span>
					)}
				</label>
				{field.type === 'number' && (
					<input
						id={fieldId}
						type="number"
						className="canvas-modal__parameter-input"
						value={currentValue ?? ''}
						onChange={(e) => {
							const val = e.target.value;
							if (val === '' && field.nullable) {
								onHyperparamChange(field.name, null);
							} else {
								const num = parseFloat(val);
								if (!isNaN(num)) {
									onHyperparamChange(field.name, num);
								}
							}
						}}
						min={field.min}
						max={field.max}
						step={field.step}
						placeholder={field.nullable ? 'Empty = default' : ''}
					/>
				)}
				{field.type === 'select' && (
					<select
						id={fieldId}
						className="canvas-modal__parameter-select"
						value={currentValue ?? ''}
						onChange={(e) => onHyperparamChange(field.name, e.target.value)}
					>
						{field.options?.map((opt) => (
							<option key={String(opt.value)} value={String(opt.value)}>
								{opt.label}
							</option>
						))}
					</select>
				)}
				{field.type === 'boolean' && (
					<input
						id={fieldId}
						type="checkbox"
						className="canvas-modal__parameter-checkbox"
						checked={Boolean(currentValue)}
						onChange={(e) => onHyperparamChange(field.name, e.target.checked)}
					/>
				)}
				{field.type === 'text' && (
					<input
						id={fieldId}
						type="text"
						className="canvas-modal__parameter-input"
						value={currentValue ?? ''}
						onChange={(e) => onHyperparamChange(field.name, e.target.value)}
						placeholder={field.nullable ? 'Empty = default' : ''}
					/>
				)}
			</div>
		);
	};

	if (!modelType && hyperparametersParameter) {
		return <>{renderParameterField(hyperparametersParameter)}</>;
	}

	if (!modelType || hyperparamFields.length === 0) {
		return null;
	}

	return (
		<>
			<div className="canvas-modal__parameter" style={{ gridColumn: '1 / -1' }}>
				<div
					style={{
						display: 'flex',
						flexWrap: 'wrap',
						alignItems: 'flex-start',
						justifyContent: 'space-between',
						gap: '1rem',
						marginBottom: '0.75rem',
						padding: '0.75rem 1rem',
						background: 'rgba(15, 23, 42, 0.35)',
						borderRadius: '6px',
						border: '1px solid rgba(148, 163, 184, 0.25)'
					}}
				>
					<div style={{ flex: '1 1 260px' }}>
						<strong style={{ fontSize: '0.95rem', color: '#e2e8f0' }}>Model Configuration</strong>
						<p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
							{showAdvanced
								? 'Fine-tune hyperparameters or use defaults for quick training'
								: 'Using default hyperparameters — toggle Advanced to customize'}
						</p>
						{(primaryPreset || latestTuningJob) && (
							<p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: 'rgba(148, 163, 184, 0.8)' }}>
								{primaryPreset ? (
									<>
										✓ Tuned parameters available for <strong>{primaryPreset.modelType}</strong>
										{primaryPreset.runNumber ? ` (run ${primaryPreset.runNumber})` : ''}
										{primaryPresetRelative ? ` • ${primaryPresetRelative}` : ''}
										{primaryPreset.targetColumn ? ` • Target: ${primaryPreset.targetColumn}` : ''}
										{primaryPresetScoreLabel ? ` • ${primaryPresetScoreLabel}` : ''}
									</>
								) : (
									<>
										Latest tuning run {latestTuningRunNumber || 'N/A'}
										{latestTuningRelative ? ` • ${latestTuningRelative}` : ''}
										{latestTuningJob?.model_type && latestTuningJob.model_type !== modelType
											? ` targeted ${latestTuningJob.model_type}.`
											: ' did not produce reusable parameters for this configuration yet.'}
									</>
								)}
							</p>
						)}
						{applyStatus && (
							<p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: applyStatusColor }}>
								{applyStatus.message}
							</p>
						)}
					</div>
					<div
						style={{
							display: 'flex',
							flexWrap: 'wrap',
							gap: '0.5rem',
							alignItems: 'center',
							justifyContent: 'flex-end'
						}}
					>
						<button
							type="button"
							onClick={onApplyBestParams}
							disabled={applyButtonDisabled}
							style={{
								padding: '0.5rem 1rem',
								fontSize: '0.875rem',
								fontWeight: 500,
								color: applyButtonDisabled ? 'rgba(148, 163, 184, 0.55)' : '#0c4a6e',
								background: applyButtonDisabled ? 'rgba(15, 23, 42, 0.35)' : 'rgba(191, 219, 254, 0.9)',
								border: applyButtonDisabled ? '1px solid rgba(148, 163, 184, 0.35)' : '1px solid rgba(147, 197, 253, 0.9)',
								borderRadius: '4px',
								cursor: applyButtonDisabled ? 'not-allowed' : 'pointer',
								transition: 'all 0.2s ease',
								minWidth: '130px',
								boxShadow: applyButtonDisabled ? 'none' : '0 3px 10px rgba(14, 116, 144, 0.25)'
							}}
						>
							Apply tuned params
						</button>
						<button
							type="button"
							onClick={onBrowsePresets}
							disabled={browsePresetsDisabled}
							style={{
								padding: '0.5rem 1rem',
								fontSize: '0.875rem',
								fontWeight: 500,
								color: browsePresetsDisabled ? 'rgba(148, 163, 184, 0.55)' : 'rgba(168, 85, 247, 0.95)',
								background: browsePresetsDisabled ? 'rgba(15, 23, 42, 0.35)' : 'rgba(76, 29, 149, 0.18)',
								border: browsePresetsDisabled ? '1px solid rgba(148, 163, 184, 0.35)' : '1px solid rgba(168, 85, 247, 0.45)',
								borderRadius: '4px',
								cursor: browsePresetsDisabled ? 'not-allowed' : 'pointer',
								transition: 'all 0.2s ease',
								minWidth: '150px'
							}}
						>
							Browse tuned params
						</button>
						<button
							type="button"
							onClick={onToggleAdvanced}
							style={{
								padding: '0.5rem 1rem',
								fontSize: '0.875rem',
								fontWeight: '500',
								color: showAdvanced ? '#fff' : 'rgba(148, 163, 184, 0.85)',
								background: showAdvanced ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(15, 23, 42, 0.6)',
								border: showAdvanced ? 'none' : '1px solid rgba(148, 163, 184, 0.45)',
								borderRadius: '4px',
								cursor: 'pointer',
								transition: 'all 0.2s ease',
								minWidth: '100px',
								boxShadow: showAdvanced ? '0 4px 12px rgba(118, 75, 162, 0.25)' : 'none'
							}}
							onMouseEnter={(e) => {
								if (showAdvanced) {
									e.currentTarget.style.boxShadow = '0 6px 16px rgba(118, 75, 162, 0.35)';
								} else {
									e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.65)';
									e.currentTarget.style.background = 'rgba(15, 23, 42, 0.8)';
								}
							}}
							onMouseLeave={(e) => {
								if (showAdvanced) {
									e.currentTarget.style.boxShadow = '0 4px 12px rgba(118, 75, 162, 0.25)';
								} else {
									e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.45)';
									e.currentTarget.style.background = 'rgba(15, 23, 42, 0.6)';
								}
							}}
						>
							{showAdvanced ? '✓ Advanced' : 'Default'}
						</button>
					</div>
				</div>
			</div>

			{isLoading && (
				<p className="canvas-modal__note canvas-modal__note--muted">Loading hyperparameters…</p>
			)}
			{error && (
				<p className="canvas-modal__note canvas-modal__note--error">
					Failed to load hyperparameters: {error.message}
				</p>
			)}
			{showAdvanced && hyperparamFields.length > 0 && (
				<div style={{ 
					gridColumn: '1 / -1',
					padding: '1.25rem',
					background: 'rgba(15, 23, 42, 0.25)',
					border: '1px solid rgba(148, 163, 184, 0.2)',
					borderRadius: '6px',
					marginTop: '0.5rem'
				}}>
					<div style={{ 
						marginBottom: '1rem',
						paddingBottom: '0.75rem',
						borderBottom: '2px solid rgba(102, 126, 234, 0.3)'
					}}>
						<h4 style={{ 
							margin: '0 0 0.5rem 0', 
							fontSize: '1rem',
							fontWeight: '600',
							color: '#e2e8f0'
						}}>
							Advanced Hyperparameters
						</h4>
						<p style={{ 
							margin: '0', 
							fontSize: '0.875rem', 
							color: 'rgba(148, 163, 184, 0.85)',
							lineHeight: '1.5'
						}}>
							Configure {modelType.replace(/_/g, ' ')} hyperparameters. Leave empty to use recommended defaults.
						</p>
					</div>
					<div style={{ 
						display: 'grid',
						gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
						gap: '1rem'
					}}>
						{hyperparamFields.map(renderHyperparamField)}
					</div>
				</div>
			)}
		</>
	);
};
